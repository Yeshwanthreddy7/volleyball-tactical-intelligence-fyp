"""Pass 3: wraps bicausal_render.py's proven video renderer and burns a live
"paper features" HUD onto every frame, so every quantity paper_features.py
computed (O, P, K, E, I, R, cls_inter, cls_rela, exchange tokens, Z, L_KL) is
visibly on screen in the output video, not just saved to .npz on disk.

render_frames() in bicausal_render.py already draws the paper's Relation (R)
and Human-Object Interaction (I) *concepts* (team graph + ball-interaction
highlight, rule-based). This script adds a second, literal layer: the actual
numeric feature vectors paper_features.py computed for the current T=10-frame
window, as a small readout box + an E heatmap thumbnail.

Usage:
  python paper_output_render.py --cache ../cache/cache_vp2_full.pkl \
      --features ../features --out "../video/videoplayback(2)_bicausal_paper.mp4"
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "OUTPUTS"))
import bicausal_render as bc  # noqa: E402


def _font(size):
    try:
        return ImageFont.truetype("segoeuib.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("arialbd.ttf", size)
        except Exception:
            return ImageFont.load_default()


def load_window_summaries(features_dir: Path):
    manifest = json.loads((features_dir / "manifest.json").read_text())
    T = manifest["window_T"]
    shards = sorted(features_dir.glob("paper_features_shard*.npz"))
    summaries = []
    for sp in shards:
        z = np.load(sp)
        window = z["window"]
        O = z["O"].astype(np.float32)
        P = z["P"].astype(np.float32)
        K = z["K"].astype(np.float32)
        E = z["E"].astype(np.float32)                    # (n, T, Nv, Nv)
        I = z["I"].astype(np.float32)
        R = z["R"].astype(np.float32)
        cls_inter = z["cls_inter"].astype(np.float32)
        cls_rela = z["cls_rela"].astype(np.float32)
        exch_inter = z["exchange_token_inter"].astype(np.float32)
        exch_rela = z["exchange_token_rela"].astype(np.float32)
        Z_final = z["Z_final"].astype(np.float32)
        L_KL = z["L_KL"]
        n = len(window)
        for k in range(n):
            summaries.append({
                "window": int(window[k]),
                "O_norm": float(np.linalg.norm(O[k])),
                "P_norm": float(np.linalg.norm(P[k])),
                "K_norm": float(np.linalg.norm(K[k])),
                "E_mean": E[k].mean(axis=0),                          # (Nv, Nv)
                "I_norm": float(np.linalg.norm(I[k])),
                "R_norm": float(np.linalg.norm(R[k])),
                "cls_inter_norm": float(np.linalg.norm(cls_inter[k])),
                "cls_rela_norm": float(np.linalg.norm(cls_rela[k])),
                "exch_inter_norm": float(np.linalg.norm(exch_inter[k])),
                "exch_rela_norm": float(np.linalg.norm(exch_rela[k])),
                "Z_norm": float(np.linalg.norm(Z_final[k])),
                "L_KL": float(L_KL[k]),
            })
        z.close()
    summaries.sort(key=lambda s: s["window"])
    return summaries, T, manifest


def heatmap_rgb(E, size=92):
    """E (Nv,Nv) float -> size x size RGB uint8 block, blue-white-red diverging."""
    m = np.abs(E).max() + 1e-6
    v = np.clip(E / m, -1, 1)
    r = np.where(v > 0, 1.0, 1.0 + v)
    g = 1.0 - np.abs(v)
    b = np.where(v < 0, 1.0, 1.0 - v)
    rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    img = Image.fromarray(rgb, "RGB").resize((size, size), Image.NEAREST)
    return np.array(img)


_F_TITLE = _font(15)
_F_SMALL = _font(13)


def draw_hud(main_rgb: np.ndarray, s: dict, box=None):
    h, w = main_rgb.shape[:2]
    if box is None:
        box = (int(w * 0.749), int(h * 0.653), w - 10, h - 4)
    x0, y0, x1, y1 = box
    # translucent panel background: plain numpy blend, same technique bicausal_render.py
    # already uses (cv2.addWeighted) rather than PIL's RGBA-draw-on-RGB-image mode
    region = main_rgb[y0:y1, x0:x1].astype(np.float32)
    panel_color = np.array([10, 10, 15], np.float32)
    main_rgb[y0:y1, x0:x1] = (region * 0.25 + panel_color * 0.75).astype(np.uint8)
    main_rgb[y0:y1, x0] = main_rgb[y0:y1, x1 - 1] = (255, 255, 255)
    main_rgb[y0, x0:x1] = main_rgb[y1 - 1, x0:x1] = (255, 255, 255)

    pil = Image.fromarray(main_rgb)
    d = ImageDraw.Draw(pil)
    f_t, f_s = _F_TITLE, _F_SMALL
    d.text((x0 + 10, y0 + 6), "Paper features (Bi-Causal)", font=f_t, fill=(255, 220, 90))
    d.text((x0 + 10, y0 + 26), f"window #{s['window']}", font=f_s, fill=(190, 190, 190))

    hm = heatmap_rgb(s["E_mean"], size=92)
    pil.paste(Image.fromarray(hm), (x0 + 10, y0 + 46))
    d.text((x0 + 10, y0 + 140), "E (graph edges)", font=f_s, fill=(190, 190, 190))

    lines = [
        f"O (ball)      ||.|| = {s['O_norm']:.2f}",
        f"P (players)   ||.|| = {s['P_norm']:.2f}",
        f"K (subgroup)  ||.|| = {s['K_norm']:.2f}",
        f"I (interact.) ||.|| = {s['I_norm']:.2f}",
        f"R (relation)  ||.|| = {s['R_norm']:.2f}",
        f"cls_inter     ||.|| = {s['cls_inter_norm']:.2f}",
        f"cls_rela      ||.|| = {s['cls_rela_norm']:.2f}",
        f"exchange tok. I->R={s['exch_inter_norm']:.2f} R->I={s['exch_rela_norm']:.2f}",
        f"Z (final fuse) ||.|| = {s['Z_norm']:.2f}",
        f"L_KL = {s['L_KL']:.3f}",
    ]
    ly = y0 + 46
    lx = x0 + 112
    for line in lines:
        d.text((lx, ly), line, font=f_s, fill=(235, 235, 235))
        ly += 17
    return np.array(pil)


def _new_writer(path, size, fps):
    writer = imageio_ffmpeg.write_frames(
        path, size, fps=fps, codec="libx264",
        quality=None, pix_fmt_out="yuv420p",
        output_params=["-crf", "16", "-preset", "medium"],
        macro_block_size=16)
    writer.send(None)
    return writer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", default=str(HERE.parent.parent / "fyp" / "labels.csv"))
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--segment-frames", type=int, default=4000,
                     help="write the video in chunks this long, concatenated losslessly at "
                          "the end -- avoids one fragile multi-hour pipe to ffmpeg")
    args = ap.parse_args()

    print("loading paper-feature window summaries ...", flush=True)
    summaries, T, manifest = load_window_summaries(Path(args.features))
    print(f"{len(summaries)} windows, T={T}", flush=True)

    out_path = Path(args.out)
    out_stem = out_path.with_suffix("")
    seg_dir = out_path.parent / (out_stem.name + "_segments")
    seg_dir.mkdir(parents=True, exist_ok=True)

    info = {}
    writer = None
    seg_paths = []
    seg_idx = 0
    frames_in_seg = 0
    last_idx = len(summaries) - 1
    for fi, main_rgb, panel_rgb in bc.render_frames(args.cache, out_stem, args.labels,
                                                      args.max_frames, info):
        w = min(fi // T, last_idx)
        main_rgb = draw_hud(main_rgb, summaries[w])
        frame_rgb = np.ascontiguousarray(np.hstack([main_rgb, panel_rgb]))
        if writer is None:
            seg_path = seg_dir / f"seg{seg_idx:04d}.mp4"
            writer = _new_writer(str(seg_path), (frame_rgb.shape[1], frame_rgb.shape[0]), info["fps"])
            seg_paths.append(seg_path)
        writer.send(frame_rgb)
        frames_in_seg += 1
        if frames_in_seg >= args.segment_frames:
            writer.close()
            print(f"  closed segment {seg_idx} ({frames_in_seg} frames)", flush=True)
            writer, frames_in_seg, seg_idx = None, 0, seg_idx + 1
    if writer is not None:
        writer.close()
        print(f"  closed segment {seg_idx} ({frames_in_seg} frames)", flush=True)

    print(f"concatenating {len(seg_paths)} segments ...", flush=True)
    list_file = seg_dir / "concat_list.txt"
    with open(list_file, "w") as f:
        for p in seg_paths:
            f.write(f"file '{p.resolve().as_posix()}'\n")
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg_exe, "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
                     "-c", "copy", "-movflags", "+faststart", str(out_path)], check=True)
    print("video ->", args.out)


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
