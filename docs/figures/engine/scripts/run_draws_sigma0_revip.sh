#!/bin/bash
# Solver measurements behind the draw-ensemble, draw-cost, sigma=0 true-route and
# reversed-Ip figures (synthetic examples only).  Run from a clean clone of the
# repository with the FIXED OpenFUSIONToolkit build, one thread per process, on a
# machine with room for the draw archives (a few hundred MB); then extract the
# small JSON and plot:
#
#   BQ_REPO=<clone> OUT=<payload dir> PY=<python> OFT_PYTHONPATH=<build>/python \
#     bash run_draws_sigma0_revip.sh
#   $PY extract_draw_data.py "$OUT" "$OUT/small"
#   $PY make_draw_figures.py "$OUT/small" <figure dir> <bouquet sha> <OFT build label>
#
# The jobs are independent; they run one after the other here (about 1.5 h on one
# core).  Draw batches: 12 draws, fixed seed, inductive-shape sigma 0.10 (the
# UncertaintyConfig default) and 0.05 (the shipped notebooks' setting).
set -uo pipefail
: "${OFT_PYTHONPATH:?set to the python dir of the fixed build}"
: "${OUT:?set to a payload dir}"
: "${BQ_REPO:?set to the clean clone}"
: "${PY:?set to the python interpreter}"
SCRIPTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
export BQ_REPO PYTHONPATH="$BQ_REPO:$OFT_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"
cd "$BQ_REPO" || exit 2
mkdir -p "$OUT/suite"
log() { echo "=== $* $(date -Is)" >> "$OUT/progress.txt"; }
log start
for s in recon imas; do for e in unified legacy; do
  $PY "$SCRIPTS/sigma0_probe.py" $s $e "$OUT/sigma0" > "$OUT/sigma0_${s}_${e}.log" 2>&1; log sigma0 $s $e rc=$?
done; done
for sg in 0.10 0.05; do for s in recon imas; do
  $PY "$SCRIPTS/draws_probe.py" $s $sg "$OUT/draws_$sg" > "$OUT/draws_${s}_${sg}.log" 2>&1; log draws $s $sg rc=$?
done; done
# the reversed-Ip identities: the solver tests' own probe output (basetemp kept)
$PY -m pytest -m solver -p no:cacheprovider -q --basetemp="$OUT/suite/tmp_revip" \
    tests/test_reversed_ip_solver.py > "$OUT/suite/revip.log" 2>&1;                       log revip rc=$?
$PY -m pytest -m solver -p no:cacheprovider -q --basetemp="$OUT/suite/tmp_revip_engine" \
    tests/test_engine_reversed_ip_gfile_solver.py > "$OUT/suite/revip_engine.log" 2>&1;   log revip_engine rc=$?
log ALL_DONE
