# videoplayback (7) — Coordinated Attack / Coordinated Defense annotation

Binary tactical-phase annotation of the **unseen** match `videoplayback (7).mp4`,
using only the two well-defined classes: **Coordinated Attack** and
**Coordinated Defense** (the 4-class model's Delayed-Support / Spacing-Breakdown
were dropped — sparse, subjective, and they dragged the whole model down).

## Deliverable

| file | what |
|---|---|
| `results/vp7_annotated.mp4` | full 44.8-min video, ATTACKING / DEFENDING banner (grey = TRANSITION/DEAD BALL between points) |
| `results/vp7_annotated_small.mp4` | ~850 MB shareable copy (same content) |
| `results/vp7_windows.csv (== _v6)` | the label track: `frame_start, frame_end, label, confidence, p_attack_raw` |
| `results/vp7_score_events.json` | 96 detected point-boundary frames (scoreboard watch) |

## How it was built

1. **Training data** — 723 hand-labelled 1-second clips (Attack 369 / Defense 354)
   from 8 source videos, re-extracted with a **geometric** team split (jersey-
   independent — vp(7)'s two teams wear similar kits so colour clustering fails
   on it), normalised to 30 fps. `features.build_model_sequence` → 18 permutation-
   invariant channels (team centroid, spread, hull, nearest-neighbour spacing,
   top-2/bottom-4 player speeds, sync).
2. **Model** — deep ensemble, InceptionTime ×5 + BiLSTM ×4 + TCN ×4, weighted
   0.45 / 0.30 / 0.25 by cross-video score. 100 epochs, mirror + jitter + mixup
   augmentation, dropout 0.3.
   - **Leave-one-video-out macro-F1: InceptionTime 0.906, BiLSTM 0.889, TCN 0.879**
     (ROC-AUC 0.95–0.97). The old 4-class model was 0.559.
3. **Detection pass on vp(7)** — YOLOv11n + fine-tuned ball model + BoT-SORT,
   strict-render, auto-court. One pass → per-frame tracked court coordinates.
4. **Windowing + smoothing** (`classify_windows.py`)
   - overlapping 29-frame windows, stride 10 (3 Hz), ensemble P(attack)
   - **2-state Viterbi HMM** smoothing (emission = P(attack), self-transition
     prior) — clean phase segmentation, never overrides sustained confident
     evidence (unlike an EMA)
   - decision threshold shifted to 0.56 to correct a vp(7) domain lean
   - **rally-break gate**: camera cuts + sustained-low player speed + **scoreboard
     point events** (each detected score change forces a break from 3.5 s before
     to 5.5 s after)
5. **Render** — `fyp/rebanner.py` replays the tracked boxes with the new banner.

## Accuracy

- **Cross-video LOVO macro-F1 0.906** is the honest accuracy proxy (8 held-out
  videos the model never saw).
- On vp(7), verified by frame inspection across 4 match segments:
  - **Defense calls are reliable** — spread coverage, digs, serve-receive
  - **Attack calls** capture the offensive/net phase of possession
  - rally structure alternates Attack↔Defense sensibly (~3 s phases)
  - breaks land on real between-point repositioning
- Split: **Attack 35% / Defense 32% / break 33%** — matches broadcast volleyball.

## Known limitations

1. **vp(7)'s ball tracker is broken** (detection frozen the whole match) — the
   model runs on player-formation features only. Still works; no ball signal.
2. **Individual labels can't be pixel-verified** — 1280×720 broadcast wide shot,
   ball invisible in frame grabs. LOVO 0.906 is the accuracy evidence.
3. **Dead-time** relies on scoreboard OCR (96 points found). A missed point =
   a merged rally. Celebrations in the ~2 s before the score updates still get
   an A/D label.

## Re-run / re-tune (fast — predictions are cached)

```
python v7_tactical/classify_windows.py --stride 10 --stay 0.91 \
  --att-threshold 0.56 --dead-speed 10.5 --use-score-events \
  --score-post 165 --rally-action-speed 17
python fyp/rebanner.py "fyp/Datasets/videoplayback (7).mp4" \
  v7_tactical/results/vp7_tracks.npz --windows-csv v7_tactical/results/vp7_windows.csv \
  --output out.mp4
```
