"""Write tests/data/ida_standard_reference.npz: what ``read_ida`` (default
``ni_source="standard"``) returned on the three public IDA-layout fixtures at
commit d311694, the branch point of the Z_eff source rule.

The reader's source is taken from git at that commit (``git show
d311694:bouquet/io/ida.py``) and executed inside the installed ``bouquet.io``
package, exactly as ``tests/test_zeff_source_rule.py::_load_base_reader``
does.  Every ``IDAProfiles`` field is stored: arrays as they are (dtype
included), ``raw_bytes`` as its SHA-256, the rest as JSON.  The test compares
the current reader against this file, so it runs without git history (a
shallow clone).  Usage, from the repository root of a full clone::

    PYTHONPATH=. python tests/data/make_ida_standard_reference.py .
"""
import dataclasses
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np

REPO = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
BASE = "d311694"
FIX = ["diiid_profs_synthetic.cdf", "diiid_profs_synthetic_pfast.cdf",
       "ida_synthetic_lmode.cdf"]
src = subprocess.run(["git", "-C", REPO, "show", f"{BASE}:bouquet/io/ida.py"],
                     check=True, capture_output=True, text=True).stdout
import bouquet.io  # noqa: E402  (the package the base module lives in)
name = "bouquet.io._ida_base_reference"
spec = importlib.util.spec_from_loader(name, loader=None)
mod = importlib.util.module_from_spec(spec)
mod.__package__ = "bouquet.io"
sys.modules[name] = mod
exec(compile(src, f"{BASE}:bouquet/io/ida.py", "exec"), mod.__dict__)

out, meta = {}, {}
for fx in FIX:
    r = mod.read_ida(os.path.join(REPO, "tests", "data", fx))
    m = {}
    for fld in dataclasses.fields(r):
        v = getattr(r, fld.name)
        if isinstance(v, np.ndarray):
            out[f"{fx}/{fld.name}"] = v
            m[fld.name] = {"kind": "array"}
        elif isinstance(v, bytes):
            m[fld.name] = {"kind": "bytes_sha256",
                           "value": hashlib.sha256(v).hexdigest()}
        elif v is None:
            m[fld.name] = {"kind": "none"}
        else:
            m[fld.name] = {"kind": type(v).__name__, "value": v}
    meta[fx] = m
out["__meta__"] = np.array(json.dumps(meta, sort_keys=True))
out["__base__"] = np.array(BASE)
np.savez_compressed(os.path.join(REPO, "tests", "data",
                                 "ida_standard_reference.npz"), **out)
print(json.dumps({k: sorted(v) for k, v in meta.items()})[:500])
