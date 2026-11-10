"""bouquet.kinrecon must stay self-contained (it will move into TokaMaker).

Every module under ``bouquet/kinrecon/`` may import only the standard library,
numpy, scipy, h5py, the Open FUSION Toolkit, and other modules of
``bouquet.kinrecon`` itself.  Imports are read with ``ast``, so nothing is
executed; relative imports are resolved against the module's package.
"""
import ast
import sys
from pathlib import Path

KINRECON = Path(__file__).resolve().parents[1] / "bouquet" / "kinrecon"
ALLOWED_THIRD_PARTY = {"numpy", "scipy", "h5py", "OpenFUSIONToolkit"}


def _module_name(path):
    rel = path.relative_to(KINRECON.parent.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return parts


def _imports(path, source=None):
    text = path.read_text() if source is None else source
    tree = ast.parse(text, filename=str(path))
    pkg = _module_name(path)
    if path.name != "__init__.py":
        pkg = pkg[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield node.lineno, a.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg[:len(pkg) - (node.level - 1)] if node.level > 1 else pkg
                if node.level - 1 > len(pkg):
                    yield node.lineno, "<relative import above the top level>"
                    continue
                mod = ".".join(base + ([node.module] if node.module else []))
            else:
                mod = node.module
            yield node.lineno, mod
        elif isinstance(node, ast.Call):
            # importlib.import_module("...") / __import__("...") with a literal
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if name in ("import_module", "__import__") and node.args and \
                    isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                yield node.lineno, node.args[0].value


def _allowed(mod):
    top = mod.split(".")[0]
    if top == "bouquet":
        return mod == "bouquet.kinrecon" or mod.startswith("bouquet.kinrecon.")
    return top in sys.stdlib_module_names or top in ALLOWED_THIRD_PARTY


def test_kinrecon_imports_only_allowed_modules():
    files = sorted(KINRECON.rglob("*.py"))
    assert files, f"no modules found under {KINRECON}"
    bad = [f"{p.relative_to(KINRECON.parent.parent)}:{line}: {mod}"
           for p in files for line, mod in _imports(p) if not _allowed(mod)]
    assert not bad, "bouquet.kinrecon imports outside its allowed set:\n" + "\n".join(bad)


def test_checker_catches_violations():
    # the resolver itself: a relative import that climbs out of kinrecon
    assert not _allowed("bouquet.mse")
    assert not _allowed("matplotlib.pyplot")
    assert _allowed("bouquet.kinrecon.interfaces")
    assert _allowed("OpenFUSIONToolkit.TokaMaker")
    assert _allowed("json")
    fake = KINRECON / "adapters" / "probe.py"      # parsed, never written
    src = ("from ...mse import x\nfrom .. import interfaces\nimport importlib\n"
           "importlib.import_module('bouquet.io')\n")
    mods = [m for _, m in _imports(fake, source=src)]
    assert mods == ["bouquet.mse", "bouquet.kinrecon", "importlib", "bouquet.io"]
    assert [m for m in mods if not _allowed(m)] == ["bouquet.mse", "bouquet.io"]
