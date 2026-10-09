"""The MSE figure probe builds a configuration the engine accepts.

``docs/figures/engine/scripts/probe_engine_mse_synthetic.py`` (run by
``run_solver_figure_measurements.sh``) used to build the IMAS factory's
LEGACY configuration and set ``reconstruction_engine="unified"`` afterwards.
The factory had already set legacy-path fields the engine never reads
(``isolate_edge_jBS=False``, ``perturb_jind_in_anchor=True``,
``jBS_baseline_mode="diff"``), so the engine's settings validation refused
every MSE measurement before reconstruction.  The probe now selects the
engine in the factory call.  Since 2026-10-07 the factories set no
engine-dependent field (resolved at ``prepare_baseline()``), so the old
construction is accepted too.

Solver-free: the configuration is built and validated, nothing is solved.
"""
import importlib.util
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_SCRIPTS = os.path.join(_REPO, "docs", "figures", "engine", "scripts")
_PROBE = os.path.join(_SCRIPTS, "probe_engine_mse_synthetic.py")


@pytest.fixture(scope="module")
def probe():
    spec = importlib.util.spec_from_file_location("_probe_engine_mse", _PROBE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _chords(n):
    return dict(R=[2.0 + 0.05 * i for i in range(n)], Z=[0.0] * n,
                tg_data=[0.01] * n, sigma=[0.004] * n)


def test_the_chords_step_configuration_passes_validation(probe, tmp_path):
    from bouquet.engine import engine_settings
    b = probe._bouquet(str(tmp_path / "mse_base"))
    g = b.config.generation
    assert g.reconstruction_engine == "unified"
    engine_settings(g)                                # must not raise


@pytest.mark.parametrize("jac", ["fd_chord", "fd_broyden"])
def test_the_fit_step_configuration_passes_validation(probe, tmp_path, jac):
    from bouquet.engine import engine_settings
    b = probe._configure_fit(probe._bouquet(str(tmp_path / f"mse_{jac}")),
                             _chords(probe.N), jac)
    g = b.config.generation
    assert "mse" in g.engine_rows and g.engine_mse_jacobian == jac
    engine_settings(g)                                # must not raise


def test_the_old_construction_is_now_accepted(probe, tmp_path):
    """The construction the probe used: the legacy configuration with the
    engine flipped afterwards.  It was refused (the factory had set the
    legacy values); since 2026-10-07 the factory sets no engine-dependent
    field, so it is accepted -- the order no longer matters
    (tests/test_engine_resolved_defaults.py).  An explicit legacy value is
    still refused."""
    import bouquet as bq
    from bouquet.engine import engine_settings
    b = bq.Bouquet.from_imas(probe._OMAS, mesh=probe._MESH, time=probe._TIME,
                             n_draws=1, nthreads=1,
                             header=str(tmp_path / "old"),
                             reconstruction_engine="legacy")
    b.config.generation.reconstruction_engine = "unified"
    engine_settings(b.config.generation)              # must not raise
    b.config.generation.isolate_edge_jBS = False
    with pytest.raises(ValueError, match="isolate_edge_jBS"):
        engine_settings(b.config.generation)


def test_no_probe_or_figure_script_flips_the_engine_after_construction():
    """No script under docs/figures/engine/scripts or tests/probes assigns
    ``.reconstruction_engine`` on an existing configuration: the engine is
    selected in the factory call (or the GenerationConfig constructor)."""
    flip = re.compile(r"\.reconstruction_engine\s*=(?!=)")
    hits = []
    for d in (_SCRIPTS, os.path.join(_HERE, "probes")):
        for name in sorted(os.listdir(d)):
            if not name.endswith(".py"):
                continue
            with open(os.path.join(d, name)) as fh:
                for k, line in enumerate(fh, 1):
                    if flip.search(line.split("#", 1)[0]):
                        hits.append(f"{name}:{k}: {line.strip()}")
    assert not hits, hits
