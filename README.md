# PINO-heat-conduction
Collection of code for physics-informed neural operator training for heat conduction 

Physics-only training (no labels in the loss) of neural operators on the
`forcing_homogeneous` benchmark: a homogeneous k = 1 unit square starting at a
uniform 300 K, heated through the left wall by 4 temporal x 4 spatial flux
families, with a fixed 300 K right wall and adiabatic top and bottom. The loss
is the Crank-Nicolson finite-volume residual of the solver that generates the
reference data. Its predecessor `diffusion_forcing_single` (varying initial
conditions) is also supported.

Ported from `no-tps-ihcp` commit `7bbad5b`, the last commit before the
physics-only stack was removed there. See [Changes from the source](#changes-from-the-source).

## Layout

| Path | Contents |
|---|---|
| `problems/` | `forcing_homogeneous`, `diffusion_forcing_single`, and `forcing`, whose sampler both build on |
| `src/physics/` | FV solvers (reference data), `fv_residual` (CN residual), `fv_preconditioner` (exact and multigrid inverses), boundary forcing, MMS verification |
| `src/operators/` | `train_pino` (physics trainer), `cvit` (`ForcingICCViT`), `fno2d`, `soap`, `ss_esoap`, `train` (config loader, optimizers, schedulers) |
| `data/` | `generate_dataset` (reference trajectories), `dataset` (splits, normalization) |
| `scripts/` | `run_train_pino` (entry point), `diagnose_fv_direct_state` (gates), `diagnose_space_time_conditioning`, `inspect_val_pairs` |
| `slurm/` | MSI launchers |
| `conf/` | `config.yaml` and `benchmark/*.yaml` |

## Formulations, models and preconditioners

All choices are dotted `key=value` overrides to `scripts/run_train_pino.py` (or
trailing arguments to the launchers).

| `physics_test.pino.residual_mode` | Formulation | Models (`physics_test.model_type`) | Preconditioners (`physics_test.pino.controls`) |
|---|---|---|---|
| `autoregressive` | One-step operator: current state and interval-average flux to the next state | `cvit`, `fno` | `raw` (none); `exact`, `mg` (spatial) |
| `time_conditioned` (requires `formulation_comparison=true`) | Arbitrary time with a hard IC, `u0 + (t/T)(net - u0)` | `cvit`, `fno` | `raw`; `exact`, `mg` (spatial); with `space_time_preconditioning=true`, `exact`, `mg` (space-time) |
| `space_time` | Arbitrary time, every CN interval of a prefix from one model input | `fno` online; both in fixed screens with a `raw` + `exact` pair | `raw`; `spatial_exact`, `spatial_mg`, `spatial_mg_detached`; `exact`, `mg` (space-time) |
| `one_step` | Arbitrary time, one random interval per update (not the one-step operator) | `cvit`, `fno` in fixed screens | `raw` + `exact` pair |

- Optimizers: `training.optimizer=Adam|AdamW|SOAP|SSESOAP`, tuned under `training.soap.*` and `training.ss_esoap.*`.
- Schedulers: `training.scheduler.type=PICViTExponential|RIGNOThreePhase|CosineWarmRestarts|StepLR`.
- FNO local-convolution branch: `model.parameters.local_branch_kernel_size=3` (0 keeps the plain FNO).

## Setup

Local (Python 3.11+):

```bash
pip install -e ".[test]"
```

MSI, once per account:

```bash
bash slurm/setup_env_msi.sh
```

This creates the conda env `pino` (override with `CONDA_ENV`), which every
launcher activates. Keep it separate from other projects: `src`, `data` and
`problems` are generic top-level package names.

## Workflow

1. **Reference data.** Supplies prescribed inputs, normalization statistics and
   fixed-screen cases; validation references are re-solved during training.

   ```bash
   python data/generate_dataset.py --benchmark forcing_homogeneous --num-sims 512 --save-dir data/forcing_homogeneous
   ```

   On MSI, `sbatch slurm/generate_data_msi.sbatch` does the same.

2. **Gates.** Every run except a formulation comparison needs a passed gate
   directory. Each takes a few minutes on a laptop CPU.

   ```bash
   python scripts/diagnose_fv_direct_state.py --output-dir runs/gates/phase0_direct_state
   python scripts/diagnose_fv_direct_state.py --mg --output-dir runs/gates/phase4_direct_state_mg
   ```

   On MSI, `sbatch slurm/generate_gates_msi.sbatch` writes both. The launchers
   read `PHASE0_DIR` and `PHASE4_DIR`, which default to these paths.

3. **Train.**
   - Time-conditioned FNO, space-time objective, fresh problems every update
     (the launcher header lists every knob):

     ```bash
     sbatch --export=ALL,CONTROL=mg,OPTIMIZER=SSESOAP slurm/train_fno_space_time_msi.sbatch
     ```

   - One-step vs arbitrary-time formulation comparison (no gate, no launcher yet):

     ```bash
     python scripts/run_train_pino.py --stage online --data-dir data/forcing_homogeneous \
         --output-dir runs/cmp_cvit_onestep_mg physics_test.pino.formulation_comparison=true
     ```

     The benchmark defaults make that a one-step CViT with the multigrid
     preconditioner. For an arbitrary-time FNO with the exact space-time
     preconditioner, add
     `physics_test.model_type=fno model.parameters.hard_right_dirichlet=true
     physics_test.pino.residual_mode=time_conditioned
     physics_test.pino.space_time_preconditioning=true 'physics_test.pino.controls=["exact"]'`.

   - `slurm/train_pcvit_mg_msi.sbatch` runs the one-step CViT gate chain (exact
     fixed screen, MG fixed screen, online MG). It is still wired to
     `diffusion_forcing_single`; its `DATA_DIR` and `NORMALIZATION_PATH`
     defaults name artifacts from `no-tps-ihcp` that are not in this repo.

4. **Inspect.** `python scripts/inspect_val_pairs.py <run>/<control>/val_pairs.csv --out <dir>`
   writes lead-time and forcing-family error plots.

Online runs checkpoint and stop cleanly on `SIGUSR1`, which the launchers
request 120 s before the time limit. Resume with
`python scripts/run_train_pino.py --resume --output-dir <run>`, or
`RESUME_RUN_DIR=<run>` for a launcher.

## Outputs

Each run directory holds `config_used.yaml` (the frozen config plus data, git
and source hashes), `source_snapshot/`, `normalization.json`,
`initial_state.pt`, `status.json` and `final_metrics.json`. Each control
subdirectory holds `train_metrics.csv`, `validation_<update>.json`,
`val_pairs.csv` and checkpoints.

## Constraints

- Gated runs require the 100 x 100 grid; `--smoke` lifts this for wiring checks only.
- `exact` and `spatial_exact` solve with SciPy SuperLU on the CPU, so a GPU run
  copies every interval to the host. The multigrid controls stay on the device.
- Each run hashes and copies the files listed in `source_paths` in
  `src/operators/train_pino.py` and calls `git rev-parse HEAD`. Keep that list
  in sync when moving files, and run from a git checkout.

## Tests

```bash
pytest
```

Run from the repository root; the suite takes about 3.5 minutes on a CPU and 5 tests need CUDA.

## Changes from the source

- The registry keeps `forcing_homogeneous`, `diffusion_forcing_single` and
  `forcing`; `conf/config.yaml` defaults to `forcing_homogeneous`.
- Removed the interfaces and source benchmarks, the supervised `diffusion`
  continuity check and the rollout module it needed, the unused FNO1d model and
  conftest fixtures, and the tests that exercised them.
  `visual/physics_plots.py` keeps only `plot_direct_state_gate`.
- MSI launchers use a dedicated conda env and gate directories under
  `runs/gates/`; `slurm/generate_gates_msi.sbatch` is new, and the data
  launcher defaults to `forcing_homogeneous`.
