#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <filesystem>
#include <sstream>
#include <regex>

// Simple 2D vector utilities
struct Vec2 {
    double x{};
    double y{};

    Vec2() = default;
    Vec2(double x_, double y_) : x(x_), y(y_) {}

    Vec2 operator+(const Vec2& o) const { return {x + o.x, y + o.y}; }
    Vec2 operator-(const Vec2& o) const { return {x - o.x, y - o.y}; }
    Vec2 operator*(double s) const { return {x * s, y * s}; }
    Vec2 operator/(double s) const { return {x / s, y / s}; }

    Vec2& operator+=(const Vec2& o) { x += o.x; y += o.y; return *this; }
    Vec2& operator-=(const Vec2& o) { x -= o.x; y -= o.y; return *this; }
    Vec2& operator*=(double s) { x *= s; y *= s; return *this; }
};

inline double dot(const Vec2& a, const Vec2& b) { return a.x*b.x + a.y*b.y; }
inline double norm2(const Vec2& a) { return dot(a, a); }
inline double norm(const Vec2& a) { return std::sqrt(norm2(a)); }
inline double cross_z(const Vec2& a, const Vec2& b) { return a.x * b.y - a.y * b.x; }

// Physical constants (SI)
constexpr double G  = 6.67430e-11;         // m^3 kg^-1 s^-2
constexpr double c  = 299792458.0;         // m s^-1
constexpr double M_sun = 1.98847e30;       // kg

struct State {
    Vec2 r; // position (m)
    Vec2 v; // velocity (m/s)
};

// Newtonian acceleration
inline Vec2 accel_newtonian(const Vec2& r, double mu) {
    const double rnorm = norm(r);
    const double inv_r3 = 1.0 / (rnorm * rnorm * rnorm + 1e-300);
    return r * (-mu * inv_r3);
}

// 1PN (Schwarzschild, test-particle) acceleration
inline Vec2 accel_1pn(const Vec2& r, const Vec2& v, double mu) {
    const double rnorm = norm(r);
    const double inv_r = 1.0 / (rnorm + 1e-300);
    const double inv_r3 = inv_r * inv_r * inv_r;
    const double v2 = norm2(v);
    const double rv = dot(r, v);

    // Newtonian part
    Vec2 aN = r * (-mu * inv_r3);

    // 1PN correction
    // a1PN = (mu / (c^2 r^3)) * [ (4 mu / r - v^2) r + 4 (r·v) v ]
    const double pref = mu / (c * c) * inv_r3;
    Vec2 a1 = r * ( (4.0 * mu * inv_r - v2) );
    a1 += v * ( 4.0 * rv );
    a1 *= pref;

    return aN + a1;
}

// Osculating orbital elements from instantaneous state (planar)
struct Elements {
    double a;      // semi-major axis
    double e;      // eccentricity
    double omega;  // argument of periapsis (rad)
    double f;      // true anomaly (rad)
    double h;      // specific angular momentum
    double E;      // specific energy
};

inline Elements elements_from_state(const Vec2& r, const Vec2& v, double mu) {
    const double rnorm = norm(r);
    const double v2    = norm2(v);
    const double hz    = cross_z(r, v);
    const double hmag  = std::abs(hz);

    // Specific energy
    const double E = 0.5 * v2 - mu / rnorm;

    // Semi-major axis (bound orbits E < 0)
    double a = std::numeric_limits<double>::quiet_NaN();
    if (E < 0.0) {
        a = -mu / (2.0 * E);
    }

    // Eccentricity vector in 2D using (v x h)/mu - r_hat
    const double inv_mu = 1.0 / mu;
    const double inv_r  = 1.0 / rnorm;
    Vec2 evec{
        v.y * hz * inv_mu - r.x * inv_r,
        -v.x * hz * inv_mu - r.y * inv_r
    };
    const double e = norm(evec);

    // Argument of periapsis: angle of eccentricity vector
    const double omega = std::atan2(evec.y, evec.x);

    // True anomaly f from evec and r
    double f = std::numeric_limits<double>::quiet_NaN();
    if (e > 1e-12) {
        const double er   = dot(evec, r);
        const double erx  = cross_z(evec, r);
        const double den  = e * rnorm + 1e-300;
        f = std::atan2(erx / den, er / den);
    }

    return Elements{a, e, omega, f, hmag, E};
}

// RK4 integrator step for a system with acceleration a(r, v)
template <typename AccelFunc>
inline void rk4_step(State& s, double dt, AccelFunc&& accel, double mu) {
    Vec2 k1_r = s.v;
    Vec2 k1_v = accel(s.r, s.v, mu);

    Vec2 k2_r = s.v + k1_v * (0.5 * dt);
    Vec2 k2_v = accel(s.r + k1_r * (0.5 * dt), s.v + k1_v * (0.5 * dt), mu);

    Vec2 k3_r = s.v + k2_v * (0.5 * dt);
    Vec2 k3_v = accel(s.r + k2_r * (0.5 * dt), s.v + k2_v * (0.5 * dt), mu);

    Vec2 k4_r = s.v + k3_v * dt;
    Vec2 k4_v = accel(s.r + k3_r * dt, s.v + k3_v * dt, mu);

    s.r += (k1_r + k2_r * 2.0 + k3_r * 2.0 + k4_r) * (dt / 6.0);
    s.v += (k1_v + k2_v * 2.0 + k3_v * 2.0 + k4_v) * (dt / 6.0);
}

// ------------ Very small JSON reader (keyed extraction) ------------
struct Config {
    double M_solar = 1.0e9;
    double a_over_Rs = 100.0;
    double e = 0.60;
    int orbits = 5;
    int save_every = 10;
    double dt_scale_peri = 0.0025;
    int steps_per_orbit = 20000;
    std::string output_csv = "orbits.csv";
    std::string start_at = "pericenter"; // or "apocenter"
    std::string model = "1PN"; // "1PN" or "Newtonian"
    bool use_explicit_state = false;
    Vec2 r0{};
    Vec2 v0{};
};

inline bool extract_number(const std::string& s, const std::string& key, double& out) {
    std::regex re("\"" + key + R"(\"\s*:\s*([-+0-9.eE]+))");
    std::smatch m;
    if (std::regex_search(s, m, re)) {
        out = std::stod(m[1]);
        return true;
    }
    return false;
}
inline bool extract_integer(const std::string& s, const std::string& key, int& out) {
    std::regex re("\"" + key + R"(\"\s*:\s*([-+]?[0-9]+))");
    std::smatch m;
    if (std::regex_search(s, m, re)) {
        out = std::stoi(m[1]);
        return true;
    }
    return false;
}
inline bool extract_string(const std::string& s, const std::string& key, std::string& out) {
    std::regex re("\"" + key + R"(\"\s*:\s*\"([^\"]*)\")");
    std::smatch m;
    if (std::regex_search(s, m, re)) {
        out = m[1];
        return true;
    }
    return false;
}
inline bool extract_vec2(const std::string& s, const std::string& key, Vec2& out) {
    std::regex re("\"" + key + R"(\"\s*:\s*\[\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\])");
    std::smatch m;
    if (std::regex_search(s, m, re)) {
        out.x = std::stod(m[1]);
        out.y = std::stod(m[2]);
        return true;
    }
    return false;
}

inline Config load_config(const std::string& path) {
    Config cfg;
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        std::cout << "Config file not found (" << path << "). Using defaults.\n";
        return cfg;
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    const std::string text = oss.str();

    extract_number(text, "M_solar", cfg.M_solar);
    extract_number(text, "a_over_Rs", cfg.a_over_Rs);
    extract_number(text, "e", cfg.e);
    extract_integer(text, "orbits", cfg.orbits);
    extract_integer(text, "save_every", cfg.save_every);
    extract_number(text, "dt_scale_peri", cfg.dt_scale_peri);
    extract_integer(text, "steps_per_orbit", cfg.steps_per_orbit);
    extract_string(text, "output_csv", cfg.output_csv);
    extract_string(text, "start_at", cfg.start_at);
    extract_string(text, "model", cfg.model);

    Vec2 r0{}, v0{};
    const bool got_r0 = extract_vec2(text, "r0", r0);
    const bool got_v0 = extract_vec2(text, "v0", v0);
    if (got_r0 && got_v0) {
        cfg.use_explicit_state = true;
        cfg.r0 = r0;
        cfg.v0 = v0;
    }
    return cfg;
}
// -------------------------------------------------------------------

int main() {
    std::cout << "Working directory: " << std::filesystem::current_path() << "\n";

    // Load configuration
    const std::string cfgPath = "config.json";
    Config cfg = load_config(cfgPath);

    // Central mass and mu
    const double M = cfg.M_solar * M_sun;
    const double mu = G * M;

    // Schwarzschild radius
    const double Rs = 2.0 * mu / (c * c);

    // Initial state
    double a = cfg.a_over_Rs * Rs;
    double e = cfg.e;

    State sN{}, sGR{};
    if (cfg.use_explicit_state) {
        sN = State{cfg.r0, cfg.v0};
        sGR = sN;
        // If explicit state provided, infer a/e for logging
        Elements el0 = elements_from_state(sN.r, sN.v, mu);
        a = el0.a;
        e = el0.e;
    } else {
        // Choose pericenter or apocenter start
        if (cfg.start_at == "apocenter") {
            const double ra = a * (1.0 + e);
            const double v_apo = std::sqrt(mu * (1.0 - e) / (a * (1.0 + e)));
            sN = State{Vec2{ra, 0.0}, Vec2{0.0, -v_apo}}; // rotate/sign as desired
        } else { // pericenter
            const double rp = a * (1.0 - e);
            const double v_peri = std::sqrt(mu * (1.0 + e) / (a * (1.0 - e)));
            sN = State{Vec2{rp, 0.0}, Vec2{0.0, v_peri}};
        }
        sGR = sN;
    }

    // Useful scalars
    const double rp = a * (1.0 - e);
    const double ra = a * (1.0 + e);
    const double T = 2.0 * M_PI * std::sqrt(a * a * a / mu);
    const double v_peri = std::sqrt(mu * (1.0 + e) / (a * (1.0 - e)));

    // Time step selection
    const double dt1 = cfg.dt_scale_peri * rp / v_peri;
    const double dt2 = T / static_cast<double>(cfg.steps_per_orbit);
    const double dt = std::min(dt1, dt2);

    // Sim horizon
    const int orbits = cfg.orbits;
    const double t_end = orbits * T;

    // Output CSV (with orbital elements)
    std::ofstream ofs(cfg.output_csv);
    if (!ofs.is_open()) {
        std::cerr << "ERROR: Failed to open " << cfg.output_csv << " for writing.\n";
        return 1;
    }
    ofs << std::setprecision(16);
    const char* header =
        "t,"
        "xN,yN,xGR,yGR,"
        "rN,rGR,"
        "aN,eN,omegaN,fN,hN,EN,"
        "aGR,eGR,omegaGR,fGR,hGR,EGR\n";
    std::cout << "CSV header: " << header;
    ofs << header;

    // Console summary
    const double delta_omega = 6.0 * M_PI * mu / (a * (1.0 - e * e) * c * c); // rad/orbit
    std::cout << "Config file: " << cfgPath << "\n";
    std::cout << "Central mass M = " << M << " kg (" << cfg.M_solar << " solar masses)\n";
    std::cout << "Schwarzschild radius Rs = " << Rs << " m\n";
    std::cout << "a = " << a << " m, e = " << e << "\n";
    std::cout << "Pericenter rp = " << rp << " m, Apocenter ra = " << ra << " m\n";
    std::cout << "T (Newtonian) = " << T << " s (" << T / (365.25*24*3600.0) << " years)\n";
    std::cout << "Pericenter speed v_peri = " << v_peri << " m/s (" << (v_peri / c) << " c)\n";
    std::cout << "1PN periapsis advance per orbit (theory): " << (delta_omega * 180.0 / M_PI) << " degrees\n";
    std::cout << "dt = " << dt << " s, total time = " << t_end << " s\n";
    std::cout << "Output CSV: " << cfg.output_csv << "\n";
    std::cout << "GR model track: " << cfg.model << "\n";
    std::cout << "Writing samples...\n";

    // Choose GR acceleration model
    auto accelN = [](const Vec2& r, const Vec2& /*v*/, double mu_) { return accel_newtonian(r, mu_); };
    auto accelGR = [&](const Vec2& r, const Vec2& v, double mu_) {
        if (cfg.model == "Newtonian") return accel_newtonian(r, mu_);
        return accel_1pn(r, v, mu_);
    };

    // Save every k steps
    const int save_every = std::max(1, cfg.save_every);

    // Integrate
    double t = 0.0;
    std::uint64_t step = 0;

    while (t <= t_end) {
        if (step % save_every == 0) {
            const double rN = norm(sN.r);
            const double rG = norm(sGR.r);
            const Elements elN  = elements_from_state(sN.r, sN.v, mu);
            const Elements elGR = elements_from_state(sGR.r, sGR.v, mu);

            ofs << t << ","
                << sN.r.x << "," << sN.r.y << ","
                << sGR.r.x << "," << sGR.r.y << ","
                << rN << "," << rG << ","
                << elN.a << "," << elN.e << "," << elN.omega << "," << elN.f << "," << elN.h << "," << elN.E << ","
                << elGR.a << "," << elGR.e << "," << elGR.omega << "," << elGR.f << "," << elGR.h << "," << elGR.E << "\n";
        }

        rk4_step(sN, dt, accelN, mu);
        rk4_step(sGR, dt, accelGR, mu);

        t += dt;
        ++step;
    }

    std::cout << "Done.\n";
    return 0;
}