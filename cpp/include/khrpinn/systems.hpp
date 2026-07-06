// khrpinn/systems.hpp — physics of the two benchmark oscillators + a
// high-accuracy double-precision RK4 reference integrator for the TRUE
// dynamics (used as the ground-truth trajectory in dependability studies).
#pragma once

#include "model.hpp"

namespace khrpinn {

// Known (hard-constrained) acceleration, float32 — must match
// common/systems.py: Duffing knows -x - 0.2v; VdP knows only -x.
inline float known_accel(System s, float x, float v) {
    if (s == System::Duffing) return -x - 0.2f * v;
    return -x;
}

// Analytical residual (what the network is supposed to have learned).
inline float true_residual(System s, float x, float v) {
    if (s == System::Duffing) return -0.3f * x * x * x;
    return (1.0f - x * x) * v;
}

// Full true dynamics, double precision (for the reference integrator).
inline void full_dynamics(System s, double x, double v, double& dx, double& dv) {
    dx = v;
    if (s == System::Duffing) dv = -x - 0.2 * v - 0.3 * x * x * x;
    else dv = -x + (1.0 - x * x) * v;
}

// One classic RK4 step of the true dynamics.
inline void rk4_step(System s, double dt, double& x, double& v) {
    double k1x, k1v, k2x, k2v, k3x, k3v, k4x, k4v;
    full_dynamics(s, x, v, k1x, k1v);
    full_dynamics(s, x + 0.5 * dt * k1x, v + 0.5 * dt * k1v, k2x, k2v);
    full_dynamics(s, x + 0.5 * dt * k2x, v + 0.5 * dt * k2v, k3x, k3v);
    full_dynamics(s, x + dt * k3x, v + dt * k3v, k4x, k4v);
    x += dt / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
    v += dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
}

// Integrate the true dynamics with RK4 at substeps of dt (default 10 substeps
// per model step). Advances (x, v) by exactly `dt`.
inline void rk4_advance(System s, double dt, int substeps, double& x, double& v) {
    const double h = dt / substeps;
    for (int i = 0; i < substeps; ++i) rk4_step(s, h, x, v);
}

}  // namespace khrpinn
