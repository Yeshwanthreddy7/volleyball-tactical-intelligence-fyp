# corner_court_detection

This folder holds the 4 corners of the full volleyball court (18 m x 9 m), detected automatically in every base frame in `../base_frame/`.

## Contents

| Path | Contents |
|---|---|
| `annotated/<video>.png` | The base frame with the detected court drawn on it. Corners are red dots with their pixel coordinates; the court outline is green; the centre and attack lines are cyan. A corner outside the picture is printed in orange and marked "off-frame". |
| `labels/<video>.json` | The full label for one frame (fields listed below) |
| `yolo_pose/<video>.txt` | A YOLO-pose keypoint label, ready for training a court-corner model. The class is 0, followed by the court box, then the 4 corners as `x y visibility` (normalised; visibility 2 = visible, 0 = outside the picture). |
| `corners.csv` | One row per video: all corner coordinates, visibility, fit error and number of court lines used |
| `_contact_sheet.png` | All 11 annotated frames on one page |

**Corner order** (camera behind the near end line): `far_left`, `far_right`, `near_right`, `near_left`.

**Court coordinates:** `far_left`=(0,0), `far_right`=(9,0), `near_right`=(9,18), `near_left`=(0,18), in metres.

**Fields in each JSON label:**
- `corners_px` and `corners_visible`: the 4 corners in pixels, and whether each one is inside the picture.
- `homography_court_m_to_px`: maps any court point in metres to a pixel.
- `lines_used`: which court lines (0, 6, 9, 12, 18 m) the fit was built on, with their pixel end points.
- `lines_not_seen`: court lines that were hidden or faint.
- Fit errors.

## Results

| Video | Court lines used (m) | Fit RMSE | Corners in the picture |
|---|---|---|---|
| usa | 6, 9, 12, 18 | 0.8 px | 4 |
| videoplayback (1) | 0, 12, 18 | 2.4 px | 2 (the camera cuts off the near end) |
| videoplayback (2) | 0, 6, 9, 12, 18 | 0.5 px | 4 |
| videoplayback (3) | 0, 6, 9, 12, 18 | 1.2 px | 4 |
| videoplayback (4) | 0, 6, 9, 12, 18 | 0.7 px | 4 |
| videoplayback (5) | 0, 6, 9, 12, 18 | 1.3 px | 4 |
| videoplayback (6) | 0, 6, 9, 12, 18 | 0.6 px | 4 |
| videoplayback (7) | 0, 9, 12, 18 | 0.9 px | 4 |
| videoplayback (8) | 0, 6, 9, 12, 18 | 1.1 px | 4 |
| videoplayback (9) | 0, 6, 9, 12, 18 | 0.9 px | 4 |
| videoplayback | 0, 6, 9, 12, 18 | 1.4 px | 4 |

**How the results were checked**
- Every corner was checked visually on zoomed crops. Each detected corner sits on the painted corner, and the fitted court model runs along the painted lines.
- For `videoplayback (7)`, the corners agree within 1–5 px with court points that were measured by hand earlier.

**Points to keep in mind**
- Some near corners are hidden behind camera equipment, in `(3)`, `(5)` and `(9)`. These corners are computed by extending the visible court lines through the obstruction.
- The near corners of `(1)` are below or beside the picture. They are computed and marked `visible: false`.
- Coordinates refer to the centre of the painted 5 cm line.

## How the corners are detected

The script is `../detect_court_corners.py`.

1. **Find the white line pixels.** A pixel counts as line paint when it is brighter than the pixels 3 px away on either side. Brightness is measured as min(R, G, B), which is high only for white, so the thin white lines stand out on beige, orange and blue floors.
2. **Fit the two sidelines.** They are the strongest steep lines, one on each side, re-fitted robustly to the line pixels.
3. **Find the court lines across the court.** The court width between the sidelines is scanned row by row, and at small tilts, for rows that are mostly covered by line paint.
4. **Work out which line is which by verification.** Any 3 parallel lines fit any 3 court positions, so a good fit alone proves nothing. Instead, every combination of 3 candidate lines with every possible labelling gives one court model. Each model is scored on how well the picture supports it:
   - Paint must lie along all five predicted court lines (0, 6, 9, 12, 18 m).
   - A line that runs on solidly past the sidelines is the net tape or an ad board, so it does not count as paint.
   - The sidelines must be painted exactly between the two end lines.
   - The camera is behind the near end, so the court must widen, and the line spacing must grow, toward the camera.
5. **Fit the final model.** The best-supported model is re-fitted on every court line it explains. The corners are where that model places (0,0), (9,0), (9,18) and (0,18).

## Running it

From `finalyearproject/`:

```bash
python BASEFRAMES/detect_court_corners.py                           # all base frames
python BASEFRAMES/detect_court_corners.py --image "BASEFRAMES/base_frame/videoplayback (7).png"
python BASEFRAMES/detect_court_corners.py --video "path/to/new_video.mp4"
```

The `--video` option is for an uploaded video. It first extracts the base frame into `BASEFRAMES/base_frame/`, then detects the corners. The new video's results are added to `corners.csv` and to the contact sheet.
