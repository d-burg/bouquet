#!/bin/bash
# Solver measurements behind the review figures (synthetic examples only).
# Run on the cluster (NOT the laptop), from a clean clone of the stack top, with the
# FIXED OpenFUSIONToolkit build (jphi_update <1/R> fix + non-finite abort).
# Single-threaded; each probe part runs in its own interpreter (OFT_env is a per-process singleton).
#
#   BQ_REPO=<clone> OUT=<payload dir on the shared home> PY=<python> OFT_PYTHONPATH=<build>/python \
#     bash run_solver_figure_measurements.sh
#
# Changes against the audit's draft (figures_draft/scripts/): the interpreter is $PY (not a bare
# python3); a LEGACY-path arm on the synthetic IDS (part imas with "_legacy": true via
# BQ_ENGINE_PROBE_GC -- the probe pops it before setting fields); the g-file-frame probe also on
# the legacy path (the default path users get); the engine MSE probe (fd_chord vs fd_broyden);
# a final extraction of the small JSON (draw profiles, g-file PRES arrays) for the laptop.
set -uo pipefail
: "${OFT_PYTHONPATH:?set to the python dir of the fixed build}"
: "${OUT:?set to a payload dir on the shared home}"
: "${BQ_REPO:?set to the clean clone}"
: "${PY:?set to the python interpreter}"
SCRIPTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
export BQ_ENGINE_PROBE_SOLVELOG=1 BQ_REPO
cd "$BQ_REPO" || exit 2
mkdir -p "$OUT"
log() { echo "=== $* $(date -Is)" >> "$OUT/progress.txt"; }
log start
$PY tests/probes/measure_engine.py "$OUT/default"  --parts recon,recon_legacy,imas,imas_q0;  log default rc=$?
BQ_ENGINE_PROBE_GC='{"_legacy": true}' \
$PY tests/probes/measure_engine.py "$OUT/imas_legacy" --parts imas;                         log imas_legacy rc=$?
BQ_ENGINE_PROBE_GC='{"separatrix_pressure": "legacy"}' \
$PY tests/probes/measure_engine.py "$OUT/sep_legacy" --parts recon,recon_legacy,imas;       log sep_legacy rc=$?
# constructed p_sep (a constant added to the fast pressure; NOT an example of the repo), both settings
BQ_ENGINE_PROBE_PSEP_ADD=2000 \
$PY tests/probes/measure_engine.py "$OUT/psep2k_offset" --parts recon,recon_legacy;        log psep2k_offset rc=$?
BQ_ENGINE_PROBE_PSEP_ADD=2000 BQ_ENGINE_PROBE_GC='{"separatrix_pressure": "legacy"}' \
$PY tests/probes/measure_engine.py "$OUT/psep2k_legacy" --parts recon,recon_legacy;        log psep2k_legacy rc=$?
for arm in "recon unified offset" "recon unified legacy" "imas unified offset" "imas unified legacy" \
           "recon legacy offset"; do
  set -- $arm
  $PY tests/probes/probe_baseline_gfile_frame.py "$OUT/gfile_frame" --source $1 --engine $2 --sep $3 \
     > "$OUT/gfile_frame_$1_$2_$3.log" 2>&1;                                                log gfile_frame $arm rc=$?
done
$PY "$SCRIPTS/probe_engine_mse_synthetic.py" "$OUT/mse" --step chords > "$OUT/mse_chords.log" 2>&1; log mse chords rc=$?
for j in fd_chord fd_broyden; do
  $PY "$SCRIPTS/probe_engine_mse_synthetic.py" "$OUT/mse" --step fit --jac $j > "$OUT/mse_$j.log" 2>&1; log mse $j rc=$?
done
$PY tests/probes/measure_engine.py "$OUT/draws" --parts draws_recon,draws_imas --draws 12 --seed 12345; log draws rc=$?
$PY "$SCRIPTS/extract_small.py" "$OUT" > "$OUT/extract.log" 2>&1;                              log extract rc=$?
log ALL_DONE
# then, on the laptop, with only the JSON copied back:
#   python make_engine_solver_figures.py <local copy of the JSON> <figure dir>
