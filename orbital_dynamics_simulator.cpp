#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <regex>
#include <sstream>
#include <string>

// ---------------- Vector utilities ----------------
struct Vec2 {
    double x{};
    double y{};

    Vec2() = default;
    Vec2(double x_, double y_) : x(x_), y(y_) {}

    Vec2 operator+(const Vec2& o) const { return {x + o.x, y + o.y}; }
    Vec2 operator-(const Vec2& o) const { return {x - o.x, y - o.y}; }
    Vec2 operator*(double s)     const { return {x * s, y * s}; }
    Vec2 operator/(double s)     const { return {x / s, y / s}; }

    Vec2& operator+=(const Vec2& o) { x += o.x; y += o.y; return *this; }
    Vec2& operator-=(const Vec2& o) { x -= o.x; y -= o.y; return *this; }
    Vec2& operator*=(double s)      { x *= s; y *= s; return *this; }
};

inline double dot(const Vec2& a, const Vec2& b) { return a.x * b.x + a.y * b.y; }
inline double norm2(const Vec2& a)              { return dot(a, a); }
inline double norm(const Vec2& a)               { return std::sqrt(norm2(a)); }
inline double cross_z(const Vec2& a, const Vec2& b) { return a.x * b.y - a.y * b.x; }

// ---------------- Physical constants (SI) ----------------
constexpr double G     = 6.67430e-11;     // m^3 kg^-1 s^-2
constexpr double c     = 299792458.0;     // m s^-1
constexpr double M_sun = 1.98847e30;      // kg

// ---------------- State and elements ----------------
struct State {
    Vec2 r;  // meters
    Vec2 v;  // m/s
};

struct Elements {
    double a;      // semi-major axis (m)
    double e;      // eccentricity
    double omega;  // argument of periapsis (rad)
    double f;      // true anomaly (rad)
    double h;      // specific angular momentum (|r x v|)
    double E;      // specific energy (Newtonian form: v^2/2 - mu/r)
};

// ---------------- Energies ----------------
inline double specific_energy_newtonian(const Vec2& r, const Vec2& v, double mu) {
    const double rn = norm(r);
    const double v2 = norm2(v);
    return 0.5 * v2 - mu / rn;
}

// First Post-Newtonian (1PN) specific energy for a test particle in Schwarzschild (per unit mass):
// E_1PN = v^2/2 − μ/r + (1/c^2) [ 3/8 v^4 + 1/2 (μ/r) v^2 + 1/2 (μ/r) v_r^2 + 1/2 (μ/r)^2 ]
inline double specific_energy_1pn(const Vec2& r, const Vec2& v, double mu) {
    const double rn = norm(r);
    const double v2 = norm2(v);
    const double vr = dot(r, v) / (rn + 1e-300);
    const double mu_over_r = mu / (rn + 1e-300);
    const double termPN = (3.0 / 8.0) * v2 * v2
                        + 0.5 * mu_over_r * v2
                        + 0.5 * mu_over_r * vr * vr
                        + 0.5 * mu_over_r * mu_over_r;
    return 0.5 * v2 - mu_over_r + termPN / (c * c);
}

// ---------------- Acceleration models ----------------
inline Vec2 accel_newtonian(const Vec2& r, double mu) {
    const double rn = norm(r);
    const double inv_r3 = 1.0 / (rn * rn * rn + 1e-300);
    return r * (-mu * inv_r3);
}

// 1PN test-particle in Schwarzschild (Cartesian 2D):
// a = a_N + (μ/(c^2 r^3)) [ (4 μ/r − v^2) r + 4 (r·v) v ]
inline Vec2 accel_1pn(const Vec2& r, const Vec2& v, double mu) {
    const double rn = norm(r);
    const double inv_r = 1.0 / (rn + 1e-300);
    const double inv_r3 = inv_r * inv_r * inv_r;
    const double v2 = norm2(v);
    const double rv = dot(r, v);

    Vec2 aN = r * (-mu * inv_r3);
    const double pref = mu / (c * c) * inv_r3;
    Vec2 corr = r * (4.0 * mu * inv_r - v2);
    corr += v * (4.0 * rv);
    corr *= pref;
    return aN + corr;
}

// ---------------- Orbital elements (planar/osculating) ----------------
inline Elements elements_from_state(const Vec2& r, const Vec2& v, double mu) {
    const double rn = norm(r);
    const double v2 = norm2(v);
    const double hz = cross_z(r, v);
    const double hmag = std::abs(hz);

    // Specific energy (Newtonian form)
    const double E = 0.5 * v2 - mu / rn;

    // Semi-major axis (for E < 0)
    double a = std::numeric_limits<double>::quiet_NaN();
    if (E < 0.0) a = -mu / (2.0 * E);

    // Eccentricity vector (planar) using e = (v x h)/mu - r_hat
    const double inv_mu = 1.0 / mu;
    const double inv_r = 1.0 / rn;
    Vec2 evec{
        v.y * hz * inv_mu - r.x * inv_r,
        -v.x * hz * inv_mu - r.y * inv_r
    };
    const double e = norm(evec);

    // Argument of periapsis
    const double omega = std::atan2(evec.y, evec.x);

    // True anomaly (from evec and r)
    double f = std::numeric_limits<double>::quiet_NaN();
    if (e > 1e-14) {
        const double er = dot(evec, r);
        const double erx = cross_z(evec, r);
        const double den = e * rn + 1e-300;
        f = std::atan2(erx / den, er / den);
    }

    return Elements{a, e, omega, f, hmag, E};
}

// ---------------- Fixed-step RK4 ----------------
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

// ---------------- Adaptive RK45 (Dormand–Prince 5(4)) ----------------
struct StepResult {
    State s_out;
    double err_norm; // normalized error (<= 1 => accept)
};

inline double norm_err(const Vec2& er, const Vec2& ev,
                       const Vec2& r_scale, const Vec2& v_scale,
                       double rel_tol, double abs_tol_r, double abs_tol_v) {
    auto norm_comp = [](double err, double scale, double rel, double abs_) {
        const double s = std::max(1.0, scale); // max(|y_i|, 1)
        const double denom = abs_ + rel * s;
        return std::abs(err) / (denom > 0 ? denom : 1.0);
    };
    const double rn = std::max(
        norm_comp(er.x, r_scale.x, rel_tol, abs_tol_r),
        norm_comp(er.y, r_scale.y, rel_tol, abs_tol_r)
    );
    const double vn = std::max(
        norm_comp(ev.x, v_scale.x, rel_tol, abs_tol_v),
        norm_comp(ev.y, v_scale.y, rel_tol, abs_tol_v)
    );
    return std::max(rn, vn);
}

template <typename AccelFunc>
inline StepResult rk45_step(const State& s, double dt, AccelFunc&& accel, double mu,
                            double rel_tol, double abs_tol_r, double abs_tol_v) {
    // Dormand–Prince 5(4) coefficients
    constexpr double a21 = 1.0 / 5.0;

    constexpr double a31 = 3.0 / 40.0;
    constexpr double a32 = 9.0 / 40.0;

    constexpr double a41 = 44.0 / 45.0;
    constexpr double a42 = -56.0 / 15.0;
    constexpr double a43 = 32.0 / 9.0;

    constexpr double a51 = 19372.0 / 6561.0;
    constexpr double a52 = -25360.0 / 2187.0;
    constexpr double a53 = 64448.0 / 6561.0;
    constexpr double a54 = -212.0 / 729.0;

    constexpr double a61 = 9017.0 / 3168.0;
    constexpr double a62 = -355.0 / 33.0;
    constexpr double a63 = 46732.0 / 5247.0;
    constexpr double a64 = 49.0 / 176.0;
    constexpr double a65 = -5103.0 / 18656.0;

    constexpr double a71 = 35.0 / 384.0;
    constexpr double a72 = 0.0;
    constexpr double a73 = 500.0 / 1113.0;
    constexpr double a74 = 125.0 / 192.0;
    constexpr double a75 = -2187.0 / 6784.0;
    constexpr double a76 = 11.0 / 84.0;

    // b (5th order)
    constexpr double b1 = 35.0 / 384.0;
    constexpr double b2 = 0.0;
    constexpr double b3 = 500.0 / 1113.0;
    constexpr double b4 = 125.0 / 192.0;
    constexpr double b5 = -2187.0 / 6784.0;
    constexpr double b6 = 11.0 / 84.0;
    constexpr double b7 = 0.0;

    // b* (4th order)
    constexpr double bs1 = 5179.0 / 57600.0;
    constexpr double bs2 = 0.0;
    constexpr double bs3 = 7571.0 / 16695.0;
    constexpr double bs4 = 393.0 / 640.0;
    constexpr double bs5 = -92097.0 / 339200.0;
    constexpr double bs6 = 187.0 / 2100.0;
    constexpr double bs7 = 1.0 / 40.0;

    auto f = [&](const State& st) {
        return std::pair<Vec2, Vec2>{st.v, accel(st.r, st.v, mu)};
    };

    auto [k1_r, k1_v] = f(s);

    State s2{ s.r + k1_r * (a21 * dt),
              s.v + k1_v * (a21 * dt) };
    auto [k2_r, k2_v] = f(s2);

    State s3{ s.r + k1_r * (a31 * dt) + k2_r * (a32 * dt),
              s.v + k1_v * (a31 * dt) + k2_v * (a32 * dt) };
    auto [k3_r, k3_v] = f(s3);

    State s4{ s.r + k1_r * (a41 * dt) + k2_r * (a42 * dt) + k3_r * (a43 * dt),
              s.v + k1_v * (a41 * dt) + k2_v * (a42 * dt) + k3_v * (a43 * dt) };
    auto [k4_r, k4_v] = f(s4);

    State s5{ s.r + k1_r * (a51 * dt) + k2_r * (a52 * dt) + k3_r * (a53 * dt) + k4_r * (a54 * dt),
              s.v + k1_v * (a51 * dt) + k2_v * (a52 * dt) + k3_v * (a53 * dt) + k4_v * (a54 * dt) };
    auto [k5_r, k5_v] = f(s5);

    State s6{ s.r + k1_r * (a61 * dt) + k2_r * (a62 * dt) + k3_r * (a63 * dt) + k4_r * (a64 * dt) + k5_r * (a65 * dt),
              s.v + k1_v * (a61 * dt) + k2_v * (a62 * dt) + k3_v * (a63 * dt) + k4_v * (a64 * dt) + k5_v * (a65 * dt) };
    auto [k6_r, k6_v] = f(s6);

    State s7{ s.r + k1_r * (a71 * dt) + k2_r * (a72 * dt) + k3_r * (a73 * dt) + k4_r * (a74 * dt) + k5_r * (a75 * dt) + k6_r * (a76 * dt),
              s.v + k1_v * (a71 * dt) + k2_v * (a72 * dt) + k3_v * (a73 * dt) + k4_v * (a74 * dt) + k5_v * (a75 * dt) + k6_v * (a76 * dt) };
    auto [k7_r, k7_v] = f(s7);

    // 5th order solution
    State s5th{
        s.r + (k1_r * (b1 * dt) + k2_r * (b2 * dt) + k3_r * (b3 * dt) + k4_r * (b4 * dt) + k5_r * (b5 * dt) + k6_r * (b6 * dt) + k7_r * (b7 * dt)),
        s.v + (k1_v * (b1 * dt) + k2_v * (b2 * dt) + k3_v * (b3 * dt) + k4_v * (b4 * dt) + k5_v * (b5 * dt) + k6_v * (b6 * dt) + k7_v * (b7 * dt))
    };

    // 4th order (embedded) for error estimate
    Vec2 r4 = s.r + (k1_r * (bs1 * dt) + k2_r * (bs2 * dt) + k3_r * (bs3 * dt) + k4_r * (bs4 * dt) + k5_r * (bs5 * dt) + k6_r * (bs6 * dt) + k7_r * (bs7 * dt));
    Vec2 v4 = s.v + (k1_v * (bs1 * dt) + k2_v * (bs2 * dt) + k3_v * (bs3 * dt) + k4_v * (bs4 * dt) + k5_v * (bs5 * dt) + k6_v * (bs6 * dt) + k7_v * (bs7 * dt));

    Vec2 er = s5th.r - r4;
    Vec2 ev = s5th.v - v4;

    // Component-wise scales based on start/end magnitudes
    Vec2 r_scale{ std::max(std::abs(s.r.x), std::abs(s5th.r.x)),
                  std::max(std::abs(s.r.y), std::abs(s5th.r.y)) };
    Vec2 v_scale{ std::max(std::abs(s.v.x), std::abs(s5th.v.x)),
                  std::max(std::abs(s.v.y), std::abs(s5th.v.y)) };

    const double en = norm_err(er, ev, r_scale, v_scale, rel_tol, abs_tol_r, abs_tol_v);
    return StepResult{ s5th, en };
}

// ---------------- Minimal JSON loader ----------------
struct Config {
    // Physics
    double M_solar = 1.0e9;
    double a_over_Rs = 100.0;
    double e = 0.60;
    std::string start_at = "pericenter"; // or "apocenter"
    std::string model    = "1PN";        // "1PN" or "Newtonian"

    // Integration horizon and output
    int         orbits      = 5;
    std::string output_csv  = "orbits.csv";
    int         save_every  = 10; // every N accepted steps

    // Fixed-step parameters (used if use_adaptive=false, and to seed dt)
    double dt_scale_peri   = 0.0025;
    int    steps_per_orbit = 20000;

    // Optional explicit initial state
    bool use_explicit_state = false;
    Vec2 r0{};
    Vec2 v0{};

    // Adaptive integrator controls
    bool   use_adaptive = false; // default to fixed-step unless enabled
    double rel_tol      = 1e-9;
    double abs_tol_r    = 1e-2;  // m
    double abs_tol_v    = 1e-5;  // m/s
    double safety       = 0.9;
    double dt_min       = 1e-6;  // s
    double dt_max       = 1e9;   // s
    int    max_rejects  = 20;
};

inline bool extract_number(const std::string& s, const std::string& key, double& out) {
    std::regex re("\"" + key + R"(\"\s*:\s*([-+0-9.eE]+))");
    std::smatch m;
    if (std::regex_search(s, m, re)) { out = std::stod(m[1]); return true; }
    return false;
}
inline bool extract_integer(const std::string& s, const std::string& key, int& out) {
    std::regex re("\"" + key + R"(\"\s*:\s*([-+]?[0-9]+))");
    std::smatch m;
    if (std::regex_search(s, m, re)) { out = std::stoi(m[1]); return true; }
    return false;
}
inline bool extract_string(const std::string& s, const std::string& key, std::string& out) {
    std::regex re("\"" + key + R"(\"\s*:\s*\"([^\"]*)\")");
    std::smatch m;
    if (std::regex_search(s, m, re)) { out = m[1]; return true; }
    return false;
}
inline bool extract_bool(const std::string& s, const std::string& key, bool& out) {
    std::regex re("\"" + key + R"(\"\s*:\s*(true|false))", std::regex::icase);
    std::smatch m;
    if (std::regex_search(s, m, re)) {
        std::string v = m[1];
        std::transform(v.begin(), v.end(), v.begin(), ::tolower);
        out = (v == "true");
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
    extract_string(text, "start_at", cfg.start_at);
    extract_string(text, "model", cfg.model);

    extract_integer(text, "orbits", cfg.orbits);
    extract_string(text, "output_csv", cfg.output_csv);
    extract_integer(text, "save_every", cfg.save_every);

    extract_number(text, "dt_scale_peri", cfg.dt_scale_peri);
    extract_integer(text, "steps_per_orbit", cfg.steps_per_orbit);

    Vec2 r0{}, v0{};
    const bool got_r0 = extract_vec2(text, "r0", r0);
    const bool got_v0 = extract_vec2(text, "v0", v0);
    if (got_r0 && got_v0) {
        cfg.use_explicit_state = true;
        cfg.r0 = r0;
        cfg.v0 = v0;
    }

    extract_bool(text, "use_adaptive", cfg.use_adaptive);
    extract_number(text, "rel_tol", cfg.rel_tol);
    extract_number(text, "abs_tol_r", cfg.abs_tol_r);
    extract_number(text, "abs_tol_v", cfg.abs_tol_v);
    extract_number(text, "safety", cfg.safety);
    extract_number(text, "dt_min", cfg.dt_min);
    extract_number(text, "dt_max", cfg.dt_max);
    extract_integer(text, "max_rejects", cfg.max_rejects);

    return cfg;
}

// ---------------- Main ----------------
int main() {
    std::cout << "Working directory: " << std::filesystem::current_path() << "\n";

    // Load configuration
    const std::string cfgPath = "config.json";
    Config cfg = load_config(cfgPath);

    // Physics
    const double M  = cfg.M_solar * M_sun;
    const double mu = G * M;
    const double Rs = 2.0 * mu / (c * c);

    // Initial conditions
    double a = cfg.a_over_Rs * Rs;
    double e = cfg.e;

    State sN{}, sGR{};
    if (cfg.use_explicit_state) {
        sN = State{cfg.r0, cfg.v0};
        sGR = sN;
        // Infer a and e for logging
        const Elements el0 = elements_from_state(sN.r, sN.v, mu);
        a = el0.a; e = el0.e;
    } else {
        if (cfg.start_at == "apocenter") {
            const double ra    = a * (1.0 + e);
            const double v_apo = std::sqrt(mu * (1.0 - e) / (a * (1.0 + e)));
            sN  = State{Vec2{ra, 0.0}, Vec2{0.0, -v_apo}};
        } else {
            const double rp     = a * (1.0 - e);
            const double v_peri = std::sqrt(mu * (1.0 + e) / (a * (1.0 - e)));
            sN  = State{Vec2{rp, 0.0}, Vec2{0.0, v_peri}};
        }
        sGR = sN;
    }

    const double rp     = a * (1.0 - e);
    const double ra     = a * (1.0 + e);
    const double T      = 2.0 * M_PI * std::sqrt(a * a * a / mu);
    const double v_peri = std::sqrt(mu * (1.0 + e) / (a * (1.0 - e)));

    // Initial dt guess
    const double dt1 = cfg.dt_scale_peri * rp / v_peri;
    const double dt2 = T / static_cast<double>(cfg.steps_per_orbit);
    double dt = std::min(dt1, dt2);
    dt = std::clamp(dt, cfg.dt_min, cfg.dt_max);

    // Output CSV
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
        "aGR,eGR,omegaGR,fGR,hGR,EGR,"
        "E1PN_GR\n";
    std::cout << "CSV header: " << header;
    ofs << header;

    // Console summary
    const double delta_omega = 6.0 * M_PI * mu / (a * (1.0 - e * e) * c * c); // rad/orbit
    std::cout << "Config file: " << cfgPath << "\n";
    std::cout << "Central mass M = " << M << " kg (" << cfg.M_solar << " solar masses)\n";
    std::cout << "Schwarzschild radius Rs = " << Rs << " m\n";
    std::cout << "a = " << a << " m, e = " << e << "\n";
    std::cout << "Pericenter rp = " << rp << " m, Apocenter ra = " << ra << " m\n";
    std::cout << "T (Keplerian) = " << T << " s (" << T / (365.25 * 24 * 3600.0) << " years)\n";
    std::cout << "Pericenter speed v_peri = " << v_peri << " m/s (" << v_peri / c << " c)\n";
    std::cout << "1PN periapsis advance per orbit (theory) = " << (delta_omega * 180.0 / M_PI) << " deg\n";
    std::cout << "Integration: " << (cfg.use_adaptive ? "adaptive RK45" : "fixed RK4")
              << ", dt initial = " << dt << " s\n";
    std::cout << "Output CSV: " << cfg.output_csv << "\n";
    std::cout << "GR model for second track: " << cfg.model << "\n";

    // Accelerations
    auto accelN = [](const Vec2& r, const Vec2& /*v*/, double mu_) { return accel_newtonian(r, mu_); };
    auto accelGR = [&](const Vec2& r, const Vec2& v, double mu_) {
        if (cfg.model == "Newtonian") return accel_newtonian(r, mu_);
        return accel_1pn(r, v, mu_);
    };

    // Row writer
    auto write_sample = [&](double t_now) {
        const double rN = norm(sN.r);
        const double rG = norm(sGR.r);

        const Elements elN  = elements_from_state(sN.r, sN.v, mu);
        const Elements elGR = elements_from_state(sGR.r, sGR.v, mu);

        const double EN    = elN.E; // Newtonian specific energy (N track)
        const double EGR   = elGR.E; // Newtonian-form specific energy, evaluated on GR track (legacy)
        const double E1PNG = specific_energy_1pn(sGR.r, sGR.v, mu); // Proper 1PN specific energy (GR track)

        ofs << t_now << ","
            << sN.r.x << "," << sN.r.y << ","
            << sGR.r.x << "," << sGR.r.y << ","
            << rN << "," << rG << ","
            << elN.a << "," << elN.e << "," << elN.omega << "," << elN.f << "," << elN.h << "," << EN << ","
            << elGR.a << "," << elGR.e << "," << elGR.omega << "," << elGR.f << "," << elGR.h << "," << EGR << ","
            << E1PNG << "\n";
    };

    // Horizon
    const double t_end = static_cast<double>(cfg.orbits) * T;

    // Integrate
    double t = 0.0;
    std::uint64_t accepted = 0;
    const int save_every = std::max(1, cfg.save_every);

    // Initial row
    write_sample(t);

    if (!cfg.use_adaptive) {
        // Fixed-step RK4
        while (t < t_end) {
            const double dt_step = std::min(dt, t_end - t);
            rk4_step(sN, dt_step, accelN, mu);
            rk4_step(sGR, dt_step, accelGR, mu);
            t += dt_step;
            if (++accepted % save_every == 0 || t >= t_end) write_sample(t);
        }
    } else {
        // Adaptive RK45
        const double order = 5.0;
        while (t < t_end) {
            double dt_try = std::min(dt, t_end - t);
            int rejects = 0;

            while (true) {
                StepResult sn = rk45_step(sN, dt_try, accelN, mu, cfg.rel_tol, cfg.abs_tol_r, cfg.abs_tol_v);
                StepResult sg = rk45_step(sGR, dt_try, accelGR, mu, cfg.rel_tol, cfg.abs_tol_r, cfg.abs_tol_v);

                const double err = std::max(sn.err_norm, sg.err_norm);
                if (err <= 1.0 || dt_try <= cfg.dt_min * 1.000001) {
                    // Accept
                    sN = sn.s_out;
                    sGR = sg.s_out;
                    t += dt_try;
                    ++accepted;

                    double factor = cfg.safety * std::pow(std::max(1e-16, err), -1.0 / order);
                    factor = std::clamp(factor, 0.5, 2.0);
                    dt = std::clamp(dt_try * factor, cfg.dt_min, cfg.dt_max);

                    if (accepted % save_every == 0 || t >= t_end) write_sample(t);
                    break;
                } else {
                    // Reject and shrink
                    double factor = cfg.safety * std::pow(std::max(1e-16, err), -1.0 / order);
                    factor = std::clamp(factor, 0.25, 0.9);
                    dt_try = std::max(cfg.dt_min, dt_try * factor);

                    if (++rejects > cfg.max_rejects) {
                        std::cerr << "ERROR: Too many step rejections. Consider loosening tolerances or increasing dt_min, or raising max_rejects." << std::endl;
                        return 2;
                    }
                }
            }
        }
    }

    std::cout << "Done.\n";
    return 0;
}
