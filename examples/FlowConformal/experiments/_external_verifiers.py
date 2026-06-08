"""Subprocess wrappers for sound verifiers we run side-by-side with ours.

These tools live in their own conda envs (different torch / numpy /
packaging pins than ours, can't coexist in one Python env). We invoke
each via :func:`subprocess.run` to its env's interpreter and read the
verdict from the tool's ``--results_file`` / ``--result_file`` artifact.

Timeout policy: we rely on a single source of truth for kill semantics —
the outer ``timeout`` shell command in :file:`run_cell.sh` (mirrors
VNN-COMP's ``run_single_instance.sh``). ``subprocess.run`` here uses no
``timeout=`` argument; the tool's own ``--timeout`` flag remains plumbed
through so the tool can do graceful internal cleanup, but the outer
shell is what guarantees the process dies. ``run_cell.sh`` writes a
``TIMEOUT`` CSV row via the runner's ``--write-timeout-row`` flag when
exit code 124 is observed, so a kill mid-subprocess does not lose its
trail.

Note we deliberately do NOT use ``start_new_session=True``: keeping the
subprocess in the same process group as the runner means an outer-shell
``timeout`` signal propagates to the child verifier as well.

Public functions:
    * :func:`run_alpha_beta_crown` — αβ-CROWN on (onnx, vnnlib).
    * :func:`run_neuralsat` — NeuralSAT on (onnx, vnnlib).

Each returns ``(verdict, wall_s, error)`` where:
    * ``verdict`` ∈ ``{'UNSAT', 'SAT', 'UNKNOWN', 'TIMEOUT', 'ERROR'}``
    * ``wall_s`` is float seconds, ``None`` for ERROR with no run.
    * ``error`` is empty string on success, else a short diagnostic.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple

# ---------- αβ-CROWN ----------

ABCROWN_PYTHON = Path(os.path.expanduser(
    '~/miniconda3/envs/alpha-beta-crown/bin/python'))
ABCROWN_REPO = Path(os.path.expanduser('~/v/other/alpha-beta-CROWN'))
ABCROWN_ENTRY = ABCROWN_REPO / 'complete_verifier' / 'abcrown.py'

# Default config — vnncomp21 ACAS Xu config has the right *flavor*
# (input-split branching + PGD pre-attack) for ACAS Xu and most
# small-network benchmarks. For image classification benchmarks
# (cifar10/cifar100/yolo/vit) we'd want a different config; pass
# ``config_yaml=...`` to override.
ABCROWN_DEFAULT_CONFIG = (
    ABCROWN_REPO / 'complete_verifier' / 'exp_configs'
    / 'vnncomp21' / 'acasxu.yaml')


# ---------- NeuralSAT ----------

NEURALSAT_PYTHON = Path(os.path.expanduser(
    '~/miniconda3/envs/neuralsat/bin/python'))
NEURALSAT_REPO = Path(os.path.expanduser('~/v/other/neuralsat'))
NEURALSAT_ENTRY = NEURALSAT_REPO / 'src' / 'main.py'


# ---------- ProbStar (StarV) ----------

PROBSTAR_PYTHON = Path(os.path.expanduser(
    '~/miniconda3/envs/starv/bin/python'))
PROBSTAR_ENTRY = (
    Path(__file__).resolve().parent / 'baselines' / 'run_probstar.py'
)


# ---------- result helpers ----------

_VERDICT_NORMALIZE = {
    'unsat': 'UNSAT',
    'sat': 'SAT',
    'unknown': 'UNKNOWN',
    'timeout': 'TIMEOUT',
    # NeuralSAT-specific verdicts. ``early_stop`` is what NeuralSAT
    # writes when its bound-propagation can't make progress and it
    # gives up without a verdict (semantically UNKNOWN — the verifier
    # reports "I can't decide"). Treating it as ERROR would mask the
    # signal we actually want for the paper: "this verifier
    # systematically fails on deep MLPs".
    'early_stop': 'UNKNOWN',
    'early_terminated': 'UNKNOWN',
    # ProbStar standalone — ``not_applicable`` is what
    # :mod:`baselines.run_probstar` writes when StarV's
    # loader can't parse the network's ops (transformer attention,
    # residual Add, Gemm, etc.). It's a determined outcome (the tool
    # ran cleanly and decided it can't apply), not a crash.
    'not_applicable': 'NOT_APPLICABLE',
}


def _read_verdict(path: Path) -> str:
    """Read the first non-empty line of ``path``, normalise to our
    verdict vocabulary. Returns ``'ERROR'`` if the file is missing or
    contains an unrecognised verdict.
    """
    if not path.exists():
        return 'ERROR'
    try:
        text = path.read_text(encoding='utf-8', errors='replace').strip()
    except Exception:
        return 'ERROR'
    if not text:
        return 'ERROR'
    first_line = text.splitlines()[0].strip().lower()
    # NeuralSAT writes "unsat\n", αβ-CROWN writes just "unsat" (no
    # newline). Both also occasionally append the verdict with a
    # trailing comma + time (NeuralSAT's CLI does this, for instance);
    # split on commas to be safe.
    head = first_line.split(',')[0].strip()
    return _VERDICT_NORMALIZE.get(head, 'ERROR')


def _run_subprocess(
    cmd: list,
    *,
    cwd: Path,
    results_file: Path,
    stderr_log_path: Optional[Path] = None,
) -> Tuple[str, Optional[float], str]:
    """Spawn ``cmd``, wait for it to exit, parse the verdict file.

    No subprocess-level timeout — the outer ``run_cell.sh`` ``timeout``
    is the single kill switch. We deliberately stay in the parent's
    process group so that signal propagates to the child too.

    On any non-trivial outcome (non-zero exit, missing/unparseable
    verdict file, or unrecognised verdict) we capture stderr+stdout
    into ``stderr_log_path`` for forensics — the returned ``err``
    string is just the last 10 lines, which the runner CSV truncates
    further. To inspect the full output, read ``stderr_log_path``.
    """
    # Allow CUDA's expandable-segments allocator to reduce
    # fragmentation — important for deep-network sound verification
    # where each BaB level allocates large transient tensors.
    env = os.environ.copy()
    env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    wall = time.time() - t0
    verdict = _read_verdict(results_file)

    stdout_text = (proc.stdout or b'').decode('utf-8', 'replace')
    stderr_text = (proc.stderr or b'').decode('utf-8', 'replace')

    # Always dump stdout+stderr to the per-instance log file when one
    # is provided AND the run wasn't a clean SAT/UNSAT — gives us
    # forensic detail for debugging at near-zero cost (the file is a
    # few KB at most).
    needs_log = (
        proc.returncode != 0
        or verdict not in ('UNSAT', 'SAT', 'UNKNOWN')
    )
    if needs_log and stderr_log_path is not None:
        try:
            stderr_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(stderr_log_path, 'w') as f:
                f.write(f'# cmd: {" ".join(str(c) for c in cmd)}\n')
                f.write(f'# cwd: {cwd}\n')
                f.write(f'# returncode: {proc.returncode}\n')
                f.write(f'# wall_s: {wall:.2f}\n')
                f.write(f'# results_file: {results_file}\n')
                f.write(f'# verdict_parsed: {verdict}\n')
                f.write('# ---------- STDOUT ----------\n')
                f.write(stdout_text)
                f.write('# ---------- STDERR ----------\n')
                f.write(stderr_text)
        except Exception:
            pass

    err = ''
    if proc.returncode != 0 or verdict == 'ERROR':
        # Pull the last 10 non-empty lines from stderr (or stdout if
        # stderr is silent — NeuralSAT writes errors to stdout
        # sometimes).
        source = stderr_text.strip() or stdout_text.strip()
        tail = [ln for ln in source.splitlines() if ln.strip()][-10:]
        err = ' | '.join(tail)[:500]
    return verdict, wall, err


def _make_results_file(prefix: str, instance_tag: str) -> Path:
    """Per-instance results file under ``/tmp``. Cleared between calls
    so absence of a verdict implies a tool crash, not a stale file.
    """
    safe_tag = instance_tag.replace('/', '_').replace(' ', '_')[:120]
    p = Path('/tmp') / f'{prefix}_{safe_tag}.txt'
    if p.exists():
        try:
            p.unlink()
        except Exception:
            pass
    return p


# ---------- αβ-CROWN ----------

def run_alpha_beta_crown(
    onnx_path: str | Path,
    vnnlib_path: str | Path,
    timeout_s: int,
    *,
    config_yaml: str | Path = ABCROWN_DEFAULT_CONFIG,
    instance_tag: str = '',
    extra_args: Optional[list] = None,
) -> Tuple[str, Optional[float], str]:
    """Run αβ-CROWN on a single (onnx, vnnlib) instance.

    Args:
        onnx_path, vnnlib_path: absolute paths to the verification
            instance.
        timeout_s: budget (seconds) — passed to αβ-CROWN's internal
            ``--timeout`` so it can clean up gracefully if it self-times
            within budget. The outer ``run_cell.sh`` is the hard kill.
        config_yaml: αβ-CROWN config file. Defaults to vnncomp21's
            ACAS Xu config (input-split + PGD); override per
            benchmark for image-classification networks.
        instance_tag: stable string used to name the per-instance
            results file under ``/tmp``. Caller should pass something
            like ``f'{benchmark}_{onnx_name}_{vnnlib_name}'``.
        extra_args: list of additional CLI args appended verbatim
            (e.g. ``['--device', 'cpu']``).

    Returns:
        ``(verdict, wall_s, error)``.
    """
    onnx_path = Path(onnx_path).resolve()
    vnnlib_path = Path(vnnlib_path).resolve()
    config_yaml = Path(config_yaml).resolve()
    out = _make_results_file('abcrown', instance_tag or onnx_path.stem)
    cmd = [
        str(ABCROWN_PYTHON), str(ABCROWN_ENTRY),
        '--config', str(config_yaml),
        '--onnx_path', str(onnx_path),
        '--vnnlib_path', str(vnnlib_path),
        '--results_file', str(out),
        '--timeout', str(int(timeout_s)),
    ]
    if extra_args:
        cmd.extend(str(a) for a in extra_args)

    safe_tag = (instance_tag or onnx_path.stem).replace('/', '_')[:120]
    stderr_log = Path('/tmp') / f'abcrown_stderr_{safe_tag}.log'
    return _run_subprocess(
        cmd, cwd=ABCROWN_REPO / 'complete_verifier',
        results_file=out,
        stderr_log_path=stderr_log,
    )


# ---------- NeuralSAT ----------

def run_neuralsat(
    onnx_path: str | Path,
    vnnlib_path: str | Path,
    timeout_s: int,
    *,
    device: str = 'cuda',
    instance_tag: str = '',
    extra_args: Optional[list] = None,
) -> Tuple[str, Optional[float], str]:
    """Run NeuralSAT on a single (onnx, vnnlib) instance.

    Args:
        onnx_path, vnnlib_path: absolute paths to the verification
            instance.
        timeout_s: budget (seconds) — passed to NeuralSAT's internal
            ``--timeout``. The outer ``run_cell.sh`` is the hard kill.
        device: ``'cuda'`` or ``'cpu'``. NeuralSAT's docs say
            ``--device {cpu,cuda}``.
        instance_tag, extra_args: same conventions as
            :func:`run_alpha_beta_crown`.

    Returns:
        ``(verdict, wall_s, error)``.
    """
    onnx_path = Path(onnx_path).resolve()
    vnnlib_path = Path(vnnlib_path).resolve()
    out = _make_results_file('neuralsat', instance_tag or onnx_path.stem)
    cmd = [
        str(NEURALSAT_PYTHON), str(NEURALSAT_ENTRY),
        '--net', str(onnx_path),
        '--spec', str(vnnlib_path),
        '--device', device,
        '--timeout', str(int(timeout_s)),
        '--result_file', str(out),
    ]
    if extra_args:
        cmd.extend(str(a) for a in extra_args)

    safe_tag = (instance_tag or onnx_path.stem).replace('/', '_')[:120]
    stderr_log = Path('/tmp') / f'neuralsat_stderr_{safe_tag}.log'
    return _run_subprocess(
        cmd, cwd=NEURALSAT_REPO,
        results_file=out,
        stderr_log_path=stderr_log,
    )


# ---------- ProbStar / StarV ----------

def run_probstar(
    onnx_path: str | Path,
    vnnlib_path: str | Path,
    timeout_s: int,
    *,
    instance_tag: str = '',
    p_filter: float = 0.0,
    lp_solver: str = 'gurobi',
    gauss_alpha: float = 2.5,
    unsafe_threshold: float = 0.05,
) -> Tuple[str, Optional[float], str, dict]:
    """Run ProbStar (StarV) on a single (onnx, vnnlib) instance via
    subprocess to the ``starv`` conda env.

    Returns ``(verdict, wall_s, error, extras)`` where ``extras`` is a
    dict with the JSON body from the standalone's results file
    (``p_min``, ``p_max``, ``threshold``, ``n_disjuncts``,
    ``coverage_*`` if present). Empty dict on parse failure or when
    the standalone never wrote its results file.

    Mirrors :func:`run_alpha_beta_crown` / :func:`run_neuralsat` for
    the verdict + wall + err return path; the extras dict carries the
    ProbStar-specific diagnostic fields the runner CSV wants to log.
    """
    import json as _json

    onnx_path = Path(onnx_path).resolve()
    vnnlib_path = Path(vnnlib_path).resolve()
    out = _make_results_file('probstar', instance_tag or onnx_path.stem)
    cmd = [
        str(PROBSTAR_PYTHON), str(PROBSTAR_ENTRY),
        '--onnx_path', str(onnx_path),
        '--vnnlib_path', str(vnnlib_path),
        '--results_file', str(out),
        '--timeout', str(int(timeout_s)),
        '--p-filter', str(p_filter),
        '--lp-solver', str(lp_solver),
        '--gauss-alpha', str(gauss_alpha),
        '--unsafe-threshold', str(unsafe_threshold),
    ]

    safe_tag = (instance_tag or onnx_path.stem).replace('/', '_')[:120]
    stderr_log = Path('/tmp') / f'probstar_stderr_{safe_tag}.log'
    verdict, wall_s, err = _run_subprocess(
        cmd, cwd=Path(__file__).resolve().parent,
        results_file=out,
        stderr_log_path=stderr_log,
    )

    # Parse the JSON body for extras. The first line is the verdict
    # word (already consumed by _read_verdict); the rest is JSON.
    extras: dict = {}
    if out.exists():
        try:
            text = out.read_text(encoding='utf-8', errors='replace')
            body = text.split('\n', 1)[1] if '\n' in text else ''
            if body.strip():
                extras = _json.loads(body)
        except Exception:
            pass
    return verdict, wall_s, err, extras
