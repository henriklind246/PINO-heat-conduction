import os
from pathlib import Path
import subprocess


PROLOGUE = '''module() { :; }
conda() { printf '%s\\0' "$@" >> "$CAPTURE.conda"; printf '\\n' >> "$CAPTURE.conda"; }
python() { printf '%s\\0' "$@" >> "$CAPTURE"; printf '\\n' >> "$CAPTURE"; }
ls() { :; }
export -f module conda python ls
exec bash "$LAUNCHER" "$@"
'''


def launch(script, tmp_path, *args, **env):
    """Run a CPU launcher with stubbed MSI tools; return (project, python calls, conda calls)."""
    script = Path(script).resolve()
    subprocess.run(['bash', '-n', script], check=True)
    project = tmp_path / 'project with spaces'
    project.mkdir(exist_ok=True)
    capture = tmp_path / 'calls'
    conda_capture = Path(f'{capture}.conda')
    for path in (capture, conda_capture):
        path.unlink(missing_ok=True)
    # A clean environment keeps the caller's BENCHMARK, CONDA_ENV, ... out of the defaults.
    base = dict(PATH=os.environ['PATH'], HOME=str(tmp_path), SLURM_JOB_ID='7', SLURM_SUBMIT_DIR=str(project),
                CAPTURE=str(capture), LAUNCHER=str(script))
    subprocess.run(['bash', '-c', PROLOGUE, 'launcher', *args], env=dict(base, **env), check=True,
                   capture_output=True)

    def records(path):
        return [line.strip('\0').split('\0') for line in path.read_text().splitlines() if line] if path.exists() else []
    return project, records(capture), records(conda_capture)


def test_data_launcher_defaults_to_the_forcing_homogeneous_training_input(tmp_path):
    project, calls, conda = launch('slurm/generate_data_msi.sbatch', tmp_path)
    assert ['activate', 'pino'] in conda
    [args] = calls
    assert args[0] == str(project / 'data/generate_dataset.py')
    assert args[args.index('--benchmark') + 1] == 'forcing_homogeneous'
    assert args[args.index('--num-sims') + 1] == '512'
    # The FNO space-time launcher reads DATA_DIR=data/forcing_homogeneous by default.
    assert Path(args[args.index('--save-dir') + 1]) == project / 'data/forcing_homogeneous'

    _, calls, conda = launch('slurm/generate_data_msi.sbatch', tmp_path, 'data/dev', '64', '--nx', '20',
                             BENCHMARK='diffusion_forcing_single', CONDA_ENV='itr')
    [args] = calls
    assert ['activate', 'itr'] in conda
    assert args[args.index('--benchmark') + 1] == 'diffusion_forcing_single'
    assert (args[args.index('--save-dir') + 1], args[args.index('--num-sims') + 1]) == ('data/dev', '64')
    assert args[-2:] == ['--nx', '20']


def test_gate_launcher_writes_the_training_launchers_default_gates(tmp_path):
    project, calls, conda = launch('slurm/generate_gates_msi.sbatch', tmp_path)
    assert ['activate', 'pino'] in conda
    assert calls == [
        ['scripts/diagnose_fv_direct_state.py', '--output-dir', str(project / 'runs/gates/phase0_direct_state')],
        ['scripts/diagnose_fv_direct_state.py', '--mg', '--output-dir',
         str(project / 'runs/gates/phase4_direct_state_mg')],
    ]


def test_gate_launcher_keeps_a_finished_gate(tmp_path):
    project = tmp_path / 'project with spaces'
    finished = project / 'runs/gates/phase0_direct_state/final_metrics.json'
    finished.parent.mkdir(parents=True)
    finished.write_text('{}')
    _, calls, _ = launch('slurm/generate_gates_msi.sbatch', tmp_path)
    assert calls == [['scripts/diagnose_fv_direct_state.py', '--mg', '--output-dir',
                      str(project / 'runs/gates/phase4_direct_state_mg')]]
