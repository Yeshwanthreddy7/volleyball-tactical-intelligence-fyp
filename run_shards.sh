#!/bin/bash
# Parallel re-extraction of the labeled clips with the fixed dual-detector pipeline.
# All shards MUST share --shuffle-seed or clips get duplicated/dropped.
# Flags below MUST match fyp/pipeline.py exactly (train/serve parity).
# --exclude-pattern drops the 130 "videoplayback_clip_NNN.mp4" (plain, no
# source suffix) clips: fps/resolution fingerprint (50.0fps/1920x1080) is an
# exact match to source (3) and a categorical mismatch to (1)/(4)
# (30.0fps/1280x720) -- treated as a likely duplicate of (3), see
# AUDIT_FINDINGS.md D5. 508 target -> ~378 clips actually extracted.
cd "$(dirname "$0")"
N=${1:-4}
mkdir -p training_csv_v3
if [ "${CLEAN:-0}" = "1" ]; then
  # Full wipe - only needed once, to purge the 44 CSVs that predated this
  # session's Phase 2/3a pipeline fixes (mixing pre-fix and post-fix CSVs in
  # one training set would silently corrupt Phase 5's "clean data" claim).
  # Already done for this dataset; CLEAN=1 ./run_shards.sh to redo from zero.
  rm -f training_csv_v3/*.csv
  SKIP_FLAG=""
else
  # Resume: skip clips whose CSV already exists (an interrupted prior run),
  # instead of re-extracting ~300 already-done clips from scratch.
  SKIP_FLAG="--skip-existing"
fi
for i in $(seq 0 $((N-1))); do
  .venv/Scripts/python.exe fyp/prepare_training_data.py "dataset/dataset" \
    --output-dir training_csv_v3 $SKIP_FLAG \
    --exclude-pattern '^videoplayback_clip_[0-9]+\.mp4$' \
    --yolo-model yolo11n.pt --ball-model fyp/volleyball_best.pt \
    --imgsz 1280 --team-split colour --auto-court --court-coords linear \
    --tracker botsort --shards "$N" --shard "$i" --shuffle-seed 0 \
    > "extract_shard_$i.log" 2>&1 &
done
wait
echo "ALL SHARDS DONE: $(ls training_csv_v3/*.csv 2>/dev/null | wc -l)/378 CSVs (508 target minus 130 excluded plain/dup-of-(3) clips)"
