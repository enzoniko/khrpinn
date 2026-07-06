"""System physics definitions (verbatim from python/iclr/kan_hrpinn_paper_bptt.py)."""

import numpy as np

SYSTEM_IDS = {"Duffing": 0, "VdP": 1}


class SystemPhysics:
    """Base class for physics systems."""

    @staticmethod
    def full_dynamics(state, t):
        raise NotImplementedError

    @staticmethod
    def known_acceleration(state):
        raise NotImplementedError

    @staticmethod
    def residual_true(state):
        raise NotImplementedError

    @staticmethod
    def name():
        raise NotImplementedError


class DuffingOscillator(SystemPhysics):
    """
    Duffing: x' = v, v' = -x - 0.2v - 0.3x^3
    Known: v' = -x - 0.2v
    Residual: v' += -0.3x^3
    """

    @staticmethod
    def full_dynamics(state, t):
        x, v = state
        dx = v
        dv = -x - 0.2 * v - 0.3 * x**3
        return [dx, dv]

    @staticmethod
    def known_acceleration(state):
        """Known physics: -x - 0.2v"""
        if isinstance(state, np.ndarray):
            x, v = state[0], state[1]
            return -x - 0.2 * v
        else:
            x, v = state[:, 0:1], state[:, 1:2]
            return -x - 0.2 * v

    @staticmethod
    def residual_true(state):
        """True residual: -0.3x^3"""
        if isinstance(state, np.ndarray):
            x = state[0]
            return -0.3 * x**3
        else:
            x = state[:, 0:1]
            return -0.3 * x**3

    @staticmethod
    def name():
        return "Duffing"


class VanDerPolOscillator(SystemPhysics):
    """
    Van der Pol: x' = v, v' = -x + (1-x^2)v
    Known: v' = -x
    Residual: v' += (1-x^2)v
    """

    @staticmethod
    def full_dynamics(state, t):
        x, v = state
        dx = v
        dv = -x + (1 - x**2) * v
        return [dx, dv]

    @staticmethod
    def known_acceleration(state):
        """Known physics: -x"""
        if isinstance(state, np.ndarray):
            x = state[0]
            return -x
        else:
            x = state[:, 0:1]
            return -x

    @staticmethod
    def residual_true(state):
        """True residual: (1-x^2)v"""
        if isinstance(state, np.ndarray):
            x, v = state[0], state[1]
            return (1 - x**2) * v
        else:
            x, v = state[:, 0:1], state[:, 1:2]
            return (1 - x**2) * v

    @staticmethod
    def name():
        return "VdP"


def get_system(name):
    name = name.lower()
    if name in ("duffing", "0"):
        return DuffingOscillator()
    if name in ("vdp", "vanderpol", "van_der_pol", "1"):
        return VanDerPolOscillator()
    raise ValueError(f"Unknown system: {name}")
