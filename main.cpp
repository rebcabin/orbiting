#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

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
// Uses classical two-body relations with the supplied mu
struct Elements {
    double a;      // semi-major axis
    double e;      // eccentricity
    double omega;  // argument of periapsis (rad), angle of eccentricity vector
    double f;      // true anomaly (rad)
    double h;      // specific angular momentum (scalar |r x v|)
    double E;      // specific energy
};

inline Elements elements_from_state(const Vec2& r, const Vec2& v, double mu) {
    const double rnorm = norm(r);
    const double v2    = norm2(v);
    const double hz    = cross_z(r, v); // out-of-plane scalar
    const double hmag  = std::abs(hz);

    // Specific energy
    const double E = 0.5 * v2 - mu / rnorm;

    // Semi-major axis (bound orbits E < 0)
    double a = std::numeric_limits<double>::quiet_NaN();
    if (E < 0.0) {
        a = -mu / (2.0 * E);
    }

    // Eccentricity vector in 2D using (v x h)/mu - r_hat
    // In 2D, (v x h) = (v_y*h, -v_x*h, 0)
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
    // cos f = (evec · r) / (e r), sin f = cross(evec, r)_z / (e r)
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
    // k's for r and v
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

int main() {
    // Central mass: 1e9 solar masses
    const double M = 1.0e9 * M_sun;
    const double mu = G * M;

    // Choose orbital elements: a = 100 Rs, e = 0.6
    const double Rs = 2.0 * mu / (c * c);
    const double a = 100.0 * Rs;       // semi-major axis (m)
    const double e = 0.60;             // eccentricity

    // Derived Newtonian quantities
    const double rp = a * (1.0 - e);   // pericenter distance
    const double ra = a * (1.0 + e);   // apocenter distance
    const double T = 2.0 * M_PI * std::sqrt(a * a * a / mu); // period

    // Initial conditions at pericenter: r = (rp, 0), v = (0, v_peri) (pure tangential)
    // v_peri = sqrt(mu * (1 + e) / (a * (1 - e)))
    const double v_peri = std::sqrt(mu * (1.0 + e) / (a * (1.0 - e)));

    State sN{Vec2{rp, 0.0}, Vec2{0.0, v_peri}};  // Newtonian
    State sGR = sN;                               // 1PN starts from same initial state

    // Time stepping: choose dt small enough to resolve pericenter passage.
    const double dt1 = 0.0025 * rp / v_peri;            // fine near pericenter
    const double dt2 = T / 20000.0;                      // adequate sampling per orbit
    const double dt = std::min(dt1, dt2);

    // Simulate multiple orbits
    const int orbits = 5;
    const double t_end = orbits * T;

    // Output CSV (now includes orbital elements)
    std::ofstream ofs("orbits.csv");
    ofs << std::setprecision(16);
    ofs << "t,"
           "xN,yN,xGR,yGR,"
           "rN,rGR,"
           "aN,eN,omegaN,fN,hN,EN,"
           "aGR,eGR,omegaGR,fGR,hGR,EGR\n";

    // Print some summary to console
    const double delta_omega = 6.0 * M_PI * mu / (a * (1.0 - e * e) * c * c); // radians per orbit (1PN prediction)
    std::cout << "Central mass M = " << M << " kg (1e9 solar masses)\n";
    std::cout << "Schwarzschild radius Rs = " << Rs << " m\n";
    std::cout << "a = " << a << " m, e = " << e << "\n";
    std::cout << "Pericenter rp = " << rp << " m, Apocenter ra = " << ra << " m\n";
    std::cout << "Orbital period (Newtonian) T = " << T << " s (" << T / (365.25*24*3600.0) << " years)\n";
    std::cout << "Pericenter speed v_peri = " << v_peri << " m/s (" << (v_peri / c) << " c)\n";
    std::cout << "1PN periapsis advance per orbit (theory): " << (delta_omega * 180.0 / M_PI) << " degrees\n";
    std::cout << "Time step dt = " << dt << " s, total time = " << t_end << " s\n";
    std::cout << "Writing samples to orbits.csv\n";

    // Integrate
    double t = 0.0;
    std::uint64_t step = 0;
    auto accelN = [](const Vec2& r, const Vec2& /*v*/, double mu_) { return accel_newtonian(r, mu_); };
    auto accelGR = [](const Vec2& r, const Vec2& v, double mu_) { return accel_1pn(r, v, mu_); };

    // Save every k steps to keep file size reasonable
    const int save_every = 10;

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
    std::cout << "Tip: Plot xN vs yN and xGR vs yGR from orbits.csv to see the GR periapsis precession.\n";
    return 0;
}