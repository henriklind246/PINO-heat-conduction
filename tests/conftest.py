import pytest

from src.physics.fv_solver_1d import FVSolver1D, Layer1D


# ---------- solver fixtures ----------

SINGLE_LAYER = [Layer1D(x_left=0.0, x_right=1.0, rho=1.0, cp=1.0, k=1.0)]

SOLVER_DEFAULTS = dict(
    a=0.0, b=1.0,
    layers=SINGLE_LAYER,
    lam_target=0.5, t_final=0.5,
    flux_f=2.0, flux_A=50.0,
    t_on=0.0, t_off=0.5, phase=0.0,
)


@pytest.fixture
def small_solver():
    """N=11 solver for fast tests."""
    return FVSolver1D(N=11, **SOLVER_DEFAULTS)
