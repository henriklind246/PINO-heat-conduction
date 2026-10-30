from __future__ import annotations

from problems.base import ProblemSpec
from problems.diffusion_forcing_single import DiffusionForcingSingleProblem
from problems.forcing import ForcingProblem
from problems.forcing_homogeneous import ForcingHomogeneousProblem

# Benchmarks register here as constructors taking a representation string.
# forcing is not physics-only: it stays registered because the two physics-only
# benchmarks build on its forcing sampler and are tested against it.
REGISTRY: dict[str, type[ProblemSpec]] = {
    "diffusion_forcing_single": DiffusionForcingSingleProblem,
    "forcing_homogeneous": ForcingHomogeneousProblem,
    "forcing": ForcingProblem,
}

# The only public representation value. Kept as an explicit axis so run
# provenance (`config_used.yaml`, published manifests) stays readable.
REPRESENTATIONS: tuple[str, ...] = ("temporal_encoder",)


def get_problem(name: str, representation: str = "temporal_encoder") -> ProblemSpec:
    try:
        cls = REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown benchmark {name!r}. Available: {sorted(REGISTRY)}"
        ) from None
    if representation not in REPRESENTATIONS:
        raise ValueError(
            f"Unknown representation {representation!r} for benchmark {name!r}. "
            f"Available: {list(REPRESENTATIONS)}"
        )
    return cls(representation)
