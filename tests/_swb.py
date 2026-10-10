"""Shared fakes of the swb tests: a toolkit whose solve_with_bootstrap takes
the given arguments, and a minimal swb config."""
from bouquet import coords
from bouquet import config as _config
from bouquet.config import BouquetConfig, GenerationConfig, ImasSource, SolverConfig

#: The solve_with_bootstrap arguments the swb method needs.
SWB_ARGS = frozenset({"x", "jphi_fixed", "p_fixed"})

#: The bootstrap keyword options of a toolkit with the internal Fortran
#: bootstrap solve (solve_with_bootstrap + TokaMaker.solve_bootstrap +
#: set_boot_ops, as on the fork / OpenFUSIONToolkit PR #271), for
#: ``bouquet.config._bootstrap_kwarg_names``.
FORK_BOOTSTRAP_NAMES = frozenset(
    "mygs ne Te ni Ti Zeff Ip_target inductive_jphi Zis scale_jBS "
    "isolate_edge_jBS psi_pad iterations diagnostic_plots parameterize_jBS "
    "use_OMFIT_sauter verbose x use_sauter_eps diagnose_bs use_python_solve "
    "coord psi_N jphi_fixed p_fixed jphi_saw ffp_prof te_prof ne_prof "
    "ti_prof ni_prof F0 pres_prof jphi_fixed_prof p_fixed_prof "
    "jphi_saw_prof djBS_tol taper_edge_jBS taper_edge_psi0 taper_edge_shape "
    "saw_q_s saw_dq saw_tol saw_relax saw_ramp saw_rule".split())


def fake_oft(monkeypatch, args=SWB_ARGS):
    """A toolkit whose solve_with_bootstrap takes ``args`` and whose
    bootstrap options are the internal-solve toolkit's
    (:data:`FORK_BOOTSTRAP_NAMES`); DIFF_BS and PIN_JPHI unset."""
    monkeypatch.setattr(coords, "_swb_params", lambda: frozenset(args))
    monkeypatch.setattr(_config, "_bootstrap_kwarg_names",
                        lambda: FORK_BOOTSTRAP_NAMES | frozenset(args))
    for env in ("DIFF_BS", "PIN_JPHI"):
        monkeypatch.delenv(env, raising=False)


def swb_cfg(source=None, solver=None, **gen):
    gen.setdefault("imas_baseline", "swb")
    gen.setdefault("kinetic_source", "ida_hybrid")
    return BouquetConfig(
        source=source or ImasSource(ids_path="dd.json", time=2.3, ida_path="x.cdf"),
        solver=solver or SolverConfig(mesh_path="m.h5"),
        output_header="S", generation=GenerationConfig(**gen))
