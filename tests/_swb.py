"""Shared fakes of the swb tests: a toolkit whose solve_with_bootstrap takes
the given arguments, and a minimal swb config."""
from bouquet import coords
from bouquet.config import BouquetConfig, GenerationConfig, ImasSource, SolverConfig

#: The solve_with_bootstrap arguments the swb method needs.
SWB_ARGS = frozenset({"x", "jphi_fixed", "p_fixed"})


def fake_oft(monkeypatch, args=SWB_ARGS):
    """A toolkit whose solve_with_bootstrap takes ``args``; DIFF_BS and
    PIN_JPHI unset."""
    monkeypatch.setattr(coords, "_swb_params", lambda: frozenset(args))
    for env in ("DIFF_BS", "PIN_JPHI"):
        monkeypatch.delenv(env, raising=False)


def swb_cfg(source=None, solver=None, **gen):
    gen.setdefault("imas_baseline", "swb")
    gen.setdefault("kinetic_source", "ida_hybrid")
    return BouquetConfig(
        source=source or ImasSource(ids_path="dd.json", time=2.3, ida_path="x.cdf"),
        solver=solver or SolverConfig(mesh_path="m.h5"),
        output_header="S", generation=GenerationConfig(**gen))
