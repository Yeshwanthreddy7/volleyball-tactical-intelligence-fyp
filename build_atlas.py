# Builds the single-file Volleyball Tactical Data Atlas (figures embedded as base64).
import base64, os, html, json

FIG = r"C:\Users\yashwanth\OneDrive\Desktop\GROUP_56\finalyearproject\figures_atlas"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures_atlas", "data_atlas.html")

AUD = {
    "all": ("all reviewers", "#6b7280"),
    "cv": ("computer vision", "#2a78d6"),
    "vb": ("volleyball", "#b98400"),
    "ml": ("machine learning", "#1b8f6a"),
    "dl": ("deep learning", "#6a4bd0"),
}

SECTIONS = [
    dict(n=1, id="data", tag="all", title="What data exists",
         intro="Every labelled example in this project is a one-second window: the court positions of six players and the ball, sampled at 30 fps for 29 frames. Three corpus generations exist because the detector was corrected twice and source videos were added over time — so each figure below names the corpus it was computed on.",
         figs=[
        ("1_dataset/01_corpus_inventory.png", ["all"],
         "training_csv/ · training_csv_v3/ · training_csv_v4/ (per-clip tracking CSVs); v7_tactical/results/vp7_frames.csv; vp7_3to8_results/",
         "The raw material at a glance. 285 / 375 / 945 labelled clips across three corpus versions, plus one full-match per-frame feature trace (80,330 rows) and a 309-window deployed-model output log with a 13-model prediction matrix over the same windows."),
        ("1_dataset/02_class_distribution.png", ["all", "ml"],
         "clip filenames across training_csv/, training_csv_v3/, training_csv_v4/",
         "Coordinated Defense and Coordinated Attack make up ~75% of every corpus. Delayed Support and Spacing Breakdown stay at 5–15% each; corpus v4 roughly doubles their absolute counts but cannot make them common."),
        ("1_dataset/03_class_by_video.png", ["ml", "vb"],
         "training_csv_v3 filenames — source-video token × class token",
         "The tactical mix differs sharply between the three source videos: Spacing Breakdown is 31% of video (1) but ~4% of video (3). Under leave-one-video-out evaluation this label-prior shift is confounded with genuine camera drift."),
        ("1_dataset/04_feature_space_pca.png", ["ml", "vb"],
         "coordination_analysis.csv — 8 measured features, 285 clips, PCA on standardised values",
         "In the plane of the interpretable coordination features the four classes overlap almost completely. The class signal in the raw measurements is weak before any model is trained — which bounds everything in section 4."),
    ]),
    dict(n=2, id="cv", tag="cv", title="Detection & tracking",
         intro="Before a tactic can be classified, the pipeline has to find six players and a fast, tiny ball in an uncalibrated broadcast frame that also contains the other team, coaches, referees and crowd. These figures quantify how well that works.",
         figs=[
        ("2_cv/05_player_funnel.png", ["cv"],
         "detector-ablation measurements (README §3.2); training_csv extraction provenance",
         "Stock YOLOv11 returns 14–23 people per broadcast frame. The auto-detected court mask and the jersey-colour team split cut that to ~5.8, against a regulation six. Feature extraction measured 4.40 players/frame on corpus v1, 5.74 on v3."),
        ("2_cv/06_ball_recall.png", ["cv"],
         "detector ablation on project footage (README §3.1)",
         "A volleyball spans ~15 px on a 720p frame. At the Ultralytics-default 640 px inference size it downscales below the smallest detection stride — 15% recall versus 77% at 1280, identical weights. A volleyball fine-tune then beats stock weights 67% vs 16% at matched resolution. Hence a dual detector: stock for players, fine-tune for the ball, both at 1280."),
        ("2_cv/07_detector_ap.png", ["cv"],
         "formal detector validation on the leak-free labelled split (README §4.3 / §15.1)",
         "Player AP@50 0.819, ball AP@50 0.543, overall mAP@50 0.681. The formal average-precision numbers confirm the player/ball split is the right call, independent of the raw detection-count argument."),
        ("2_cv/08_court_trajectories.png", ["cv", "vb"],
         "training_csv_v3 — one representative clip per class, raw p1..p6 and ball columns",
         "The actual classifier input, drawn out: six position tracks over ~1 second plus a sparse ball track. Coordinated Attack shows a fast diagonal ball and a spread formation; Coordinated Defense is compact and slow; the minority classes are visually ambiguous."),
        ("2_cv/09_broadcast_detection_timeline.png", ["cv"],
         "v7_tactical/results/vp7_frames.csv (80,330 frames) · vp7_score_events.json",
         "Detection reliability across a full 45-minute broadcast. The ball is found in 99.8% of frames; 7.2 players are tracked on average because both teams stay inside the broadcast crop. Red marks are scoreboard-change events, a proxy for rally boundaries."),
    ]),
    dict(n=3, id="volleyball", tag="vb", title="Tactics & biomechanics",
         intro="These figures translate the tracked coordinates into the vocabulary a coach or sports scientist uses — approach speed, spacing, synchrony, role kinematics — and show what the deployed model actually produces for a match.",
         figs=[
        ("3_volleyball/11_biomech_by_class.png", ["vb"],
         "coordination_analysis.csv — 285 clips",
         "The human-interpretable quantities behind the model, by tactical class. Coordinated Attack is faster and more spread; Coordinated Defense is slower and tighter; Delayed Support and Spacing Breakdown sit between the two on almost every axis."),
        ("3_volleyball/12_role_velocities.png", ["vb"],
         "role_analysis.csv — roles assigned by court position + motion",
         "Peak speed by inferred role and class. Attacker approach speed is the clearest single separator between Attack and everything else; setter and libero samples are sparse, and libero is absent from the Delayed-Support clips."),
        ("3_volleyball/13_tactical_fingerprint.png", ["vb", "ml"],
         "coordination_analysis.csv — per-clip z-scores, averaged by class",
         "Each class's standardised feature profile on one grid. Attack = high speed and large hull area; Spacing Breakdown = collapsed hull; Delayed Support has almost no distinctive signature — which is why no model learns it."),
        ("3_volleyball/14_attack_vs_defense.png", ["vb", "ml"],
         "coordination_analysis.csv (Attack/Defense clips) · v7_tactical/results/lovo_binary_colour.json",
         "The separable sub-problem. Restricted to Attack versus Defense, a dedicated binary classifier reaches ROC-AUC 0.955 and macro-F1 0.90 under 8-fold leave-one-video-out — far above the full four-class model. The coordination features carry that distinction cleanly; they do not carry the others."),
        ("3_volleyball/15_rally_timeline_vp7.png", ["vb", "all"],
         "vp7_3to8_results/vp7_3to8_HGB_windows.csv — 309 one-second windows",
         "The deployed product: a per-second tactical label for a five-minute segment, with calibrated confidence and normalised entropy underneath. Confidence dips and entropy spikes at rally transitions, where the window straddles two states."),
        ("3_volleyball/16_match_tactical_budget.png", ["vb"],
         "vp7_frames.csv — per-frame predicted state over the full broadcast",
         "Where the match time goes: 47% Coordinated Defense, 29% Coordinated Attack, 13% unclassified (dead ball / transition), 10% Delayed Support, 1.4% Spacing Breakdown."),
    ]),
    dict(n=4, id="ml", tag="ml", title="Model comparison",
         intro="Nine classical models and a rule engine were scored under two protocols: stratified k-fold (clips from one video on both sides) and leave-one-video-out (every clip of the held-out video unseen). The gap between them is the headline result of the project.",
         figs=[
        ("4_ml/17_kfold_vs_lovo.png", ["ml"],
         "ml_results/ml_results.json — 9 classical models, corpus v3 / 285 clips",
         "Every model loses 0.10–0.20 macro-F1 moving from within-video k-fold to cross-video leave-one-video-out. That gap is the amount each model relied on recognising which of the three videos it was looking at."),
        ("4_ml/18_train_vs_test_all.png", ["ml", "dl"],
         "model_comparison/trainvstest_metrics.csv — 13 models, corpus v4 / 945 clips",
         "Train macro-F1 versus cross-video test. Every tree ensemble fits the training set perfectly (F1 = 1.00) and still generalises no better than the deep models, which overfit far less and score the same. Capacity is not the bottleneck."),
        ("4_ml/19_perclass_f1_heatmap.png", ["ml", "dl"],
         "temporal_bench/bench_out/METRICS.md — corpus v3, pooled leave-one-video-out",
         "Per-class F1 for every model, classical and deep. The Delayed Support column is ~0 everywhere and Spacing Breakdown is under 0.1. This is uniform across architectures because it is a data problem, not a modelling one."),
        ("4_ml/20_confusion_hgb_inception.png", ["ml", "dl"],
         "temporal_bench/bench_out/sklearn_hgb.json · deep_results_v4/deep_results.json",
         "Row-normalised confusion for the deployed gradient-boosting model (corpus v3) and the strongest deep arm, InceptionTime (corpus v4). Minority-class clips are absorbed into Attack and Defense; the deep model recovers Delayed and Spacing partially, at the cost of more Attack/Defense confusion."),
        ("4_ml/21_channel_importance.png", ["ml"],
         "ml_results/ml_results.json — channel_importance, permutation on the held-out video",
         "Permutation importance under leave-one-video-out. No single feature channel dominates; formation shape (centroid height, convex-hull area) and the tightest/loosest pair distances lead. Removing any one channel costs at most ~0.03 macro-F1."),
        ("4_ml/22_calibration.png", ["ml"],
         "ml_results/ml_results.json — confidence_uncalibrated / confidence_calibrated",
         "The raw model reports greater than 0.99 confidence on 100% of clips (mean predictive entropy 0.0003 bits of a possible 2.0). Sigmoid calibration brings mean confidence to 0.67 and entropy to 1.36 bits — a usable signal for a downstream accept-or-investigate decision."),
        ("4_ml/23_minority_class_per_fold.png", ["ml"],
         "clip counts per class per source video, all three corpora",
         "Why two classes never train. Under leave-one-video-out, Delayed Support gives a learner roughly 5–13 clips per fold spread over 3–6 videos. Below about ten examples per fold no model class evaluated recovers a category."),
    ]),
    dict(n=5, id="dl", tag="dl", title="Deep temporal models",
         intro="Five sequence architectures — Mamba (SSM), Transformer, TCN, BiLSTM and InceptionTime — were trained on the identical (29, 18) tensor under one shared hyper-parameter budget, then held to a pre-registered bar against the deployed gradient-boosting model.",
         figs=[
        ("5_dl/24_training_curves.png", ["dl"],
         "temporal_bench/bench_out/hist/*.csv — 5 architectures × 3 folds × 2 seeds",
         "Validation macro-F1 per epoch. Every arm plateaus early, near or below the gradient-boosting baseline, while training accuracy keeps rising — the classic small-data overfitting signature. (The deep validation metric is the trainer's own 5-class macro-F1; the baseline line is approximate.)"),
        ("5_dl/25_params_vs_f1.png", ["dl"],
         "temporal_bench/bench_out/SUMMARY.md",
         "Trainable parameters against cross-video macro-F1. Each deep arm carries 54k–208k parameters for ~250 training clips per fold — a parameter-to-sample ratio of 270:1 to 1033:1. Cross-video generalisation is bounded by data volume, not model capacity."),
        ("5_dl/26_deep_vs_baseline.png", ["dl"],
         "temporal_bench/bench_out/METRICS.md — pooled leave-one-video-out, both seeds",
         "The pre-registered test: a deep arm wins only if its per-fold LOVO mean beats the sklearn baseline (0.330) by more than its own per-fold standard deviation, at both seeds. None of the five architectures clears the bar at either seed."),
        ("5_dl/27_seed_sensitivity.png", ["dl"],
         "temporal_bench/bench_out/METRICS.md — seed 42 vs seed 7",
         "Two-seed repeat as the guard against a lucky initialisation. The Transformer collapses at seed 7 (it routes 42% of clips to the reject head); BiLSTM is the most seed-stable deep arm and comes closest to the baseline."),
        ("5_dl/28_model_agreement_vp7.png", ["dl", "ml"],
         "vp7_3to8_results/vp7_3to8_all_models.csv — 13 models × 309 held-out windows",
         "Every model's prediction on the same unseen broadcast segment. The classical-ML rows agree closely window-to-window; the deep rows (BiLSTM, InceptionTime, Mamba, TCN) are visibly noisier and disagree more often with the ensemble majority."),
    ]),
]

SOURCE_ROWS = [
    ("training_csv/ · _v3/ · _v4/", "285 · 375 · 945 clips", "per-clip 29-frame tracking — 6 players + ball, one CSV per labelled clip"),
    ("coordination_analysis.csv", "285 rows", "measured biomechanical / coordination summary per clip"),
    ("role_analysis.csv", "285 rows", "inferred per-role kinematics (attacker / setter / libero / defender) per clip"),
    ("v7_tactical/results/vp7_frames.csv", "80,330 rows", "full broadcast — per-frame detections, 18-dim feature vector, window prediction"),
    ("vp7_3to8_results/vp7_3to8_HGB_windows.csv", "309 rows", "deployed-model output — label, calibrated confidence, entropy, features, class probabilities"),
    ("vp7_3to8_results/vp7_3to8_all_models.csv", "309 × 13", "every model's prediction + confidence on the same windows"),
    ("ml_results/ml_results.json", "9 models", "classical-ML comparison — k-fold + LOVO, calibration, channel importance, gold set"),
    ("model_comparison/trainvstest_metrics.csv", "13 models", "train vs leave-one-video-out macro-F1 and accuracy"),
    ("deep_results_v4/deep_results.json + logs/", "4 architectures", "deep LOVO results, per-class F1, confusion, per-epoch histories"),
    ("temporal_bench/bench_out/", "5 arch × 2 seeds", "pre-registered deep-vs-sklearn benchmark — hist/, METRICS.md, SUMMARY.md"),
    ("v7_tactical/results/lovo_binary_colour.json", "8 folds", "binary Attack-vs-Defense leave-one-video-out (ROC-AUC 0.955)"),
    ("fyp/labels.csv", "159 rows", "human tactical annotations — (video_file, anchor_time, tactic_class)"),
]


def b64(path):
    with open(os.path.join(FIG, path), "rb") as f:
        return base64.b64encode(f.read()).decode()


def chip(a):
    label, col = AUD[a]
    return f'<span class="chip" style="--chip:{col}">{label}</span>'


parts = []
parts.append("""<title>Volleyball Tactical Data Atlas</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;700;800&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root{
  --bg:#eef0ed; --surface:#ffffff; --surface-2:#f6f7f4;
  --ink:#191c1b; --ink-soft:#585f59; --ink-faint:#8b918a;
  --line:#d6dad3; --line-strong:#c2c8bf;
  --accent:#15489e; --accent-soft:#e7edf7;
  --shadow:0 1px 2px rgba(20,28,20,.04),0 8px 24px rgba(20,28,20,.05);
  color-scheme:light;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#121514; --surface:#1b1f1d; --surface-2:#20241f;
  --ink:#e9ece6; --ink-soft:#a1a9a1; --ink-faint:#767d75;
  --line:#2b312c; --line-strong:#3a423b;
  --accent:#6ea6f0; --accent-soft:#1b2b40;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35);
  color-scheme:dark;
}}
:root[data-theme="dark"]{
  --bg:#121514; --surface:#1b1f1d; --surface-2:#20241f;
  --ink:#e9ece6; --ink-soft:#a1a9a1; --ink-faint:#767d75;
  --line:#2b312c; --line-strong:#3a423b;
  --accent:#6ea6f0; --accent-soft:#1b2b40;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35);
  color-scheme:dark;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  line-height:1.6;-webkit-font-smoothing:antialiased}
.mono{font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace}

nav{position:sticky;top:0;z-index:20;background:color-mix(in srgb,var(--bg) 88%,transparent);
  backdrop-filter:blur(8px);border-bottom:1px solid var(--line)}
nav .inner{max-width:1160px;margin:0 auto;padding:11px 24px;display:flex;gap:18px;align-items:baseline;flex-wrap:wrap}
nav .brand{font-family:"Archivo",sans-serif;font-weight:800;letter-spacing:-.01em;font-size:14px;margin-right:4px}
nav a{color:var(--ink-soft);text-decoration:none;font-size:12.5px;font-weight:500;
  font-family:"IBM Plex Mono",monospace}
nav a:hover{color:var(--accent)}
nav a .n{color:var(--ink-faint)}

.wrap{max-width:1160px;margin:0 auto;padding:0 24px}
.prose{max-width:64ch}

header{padding:64px 0 40px}
.eyebrow{font-family:"IBM Plex Mono",monospace;font-size:12px;font-weight:500;letter-spacing:.14em;
  text-transform:uppercase;color:var(--accent)}
h1{font-family:"Archivo",sans-serif;font-weight:800;font-size:clamp(30px,5vw,46px);
  line-height:1.06;letter-spacing:-.02em;margin:.5rem 0 .6rem;text-wrap:balance;max-width:15ch}
.dek{font-size:17px;color:var(--ink-soft);max-width:60ch;margin:0}

.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--line);
  border:1px solid var(--line);border-radius:10px;overflow:hidden;margin:34px 0 8px}
.stat{background:var(--surface);padding:18px 20px}
.stat .v{font-family:"Archivo",sans-serif;font-weight:700;font-size:24px;letter-spacing:-.01em;
  font-variant-numeric:tabular-nums}
.stat .k{font-size:12px;color:var(--ink-soft);margin-top:2px}
@media(max-width:720px){.stats{grid-template-columns:repeat(2,1fr)}}

.legend{display:flex;gap:10px;flex-wrap:wrap;margin:26px 0 4px;font-size:12.5px;color:var(--ink-soft)}
.chip{display:inline-flex;align-items:center;gap:6px;font-family:"IBM Plex Mono",monospace;
  font-size:11px;letter-spacing:.02em;padding:2px 8px;border-radius:999px;
  border:1px solid color-mix(in srgb,var(--chip) 45%,var(--line));
  color:color-mix(in srgb,var(--chip) 72%,var(--ink));
  background:color-mix(in srgb,var(--chip) 12%,var(--surface))}
.chip::before{content:"";width:6px;height:6px;border-radius:2px;background:var(--chip)}

.sources{margin:44px 0 8px;border:1px solid var(--line);border-radius:10px;overflow:hidden;background:var(--surface)}
.sources h3{margin:0;padding:14px 18px;font-family:"Archivo",sans-serif;font-size:13px;font-weight:700;
  letter-spacing:.02em;text-transform:uppercase;color:var(--ink-soft);border-bottom:1px solid var(--line);background:var(--surface-2)}
.sources table{width:100%;border-collapse:collapse;font-size:13px}
.sources td{padding:9px 18px;border-bottom:1px solid var(--line);vertical-align:top}
.sources tr:last-child td{border-bottom:none}
.sources td.f{font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--ink);white-space:nowrap}
.sources td.r{font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--accent);white-space:nowrap;text-align:right;font-variant-numeric:tabular-nums}
.sources td.d{color:var(--ink-soft)}
.sources .scroll{overflow-x:auto}
@media(max-width:720px){.sources td.d{display:none}}

section{padding:56px 0 8px;scroll-margin-top:60px;border-top:1px solid var(--line);margin-top:44px}
section:first-of-type{border-top:none}
.s-head{display:flex;gap:16px;align-items:baseline;flex-wrap:wrap;margin-bottom:6px}
.s-num{font-family:"Archivo",sans-serif;font-weight:800;font-size:15px;color:var(--accent);
  font-variant-numeric:tabular-nums}
h2{font-family:"Archivo",sans-serif;font-weight:700;font-size:clamp(22px,3.2vw,30px);
  letter-spacing:-.015em;margin:0;text-wrap:balance}
.s-intro{color:var(--ink-soft);max-width:64ch;margin:10px 0 0;font-size:15px}

.fig{margin:34px 0;background:var(--surface);border:1px solid var(--line);border-radius:12px;
  box-shadow:var(--shadow);overflow:hidden}
.fig .imgwrap{background:#fff;padding:14px;overflow-x:auto}
.fig img{display:block;width:100%;min-width:640px;height:auto}
.fig figcaption{padding:16px 20px 18px;border-top:1px solid var(--line)}
.fig .caphead{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;flex-wrap:wrap}
.fig .src{font-family:"IBM Plex Mono",monospace;font-size:11.5px;color:var(--ink-faint);
  letter-spacing:.01em;word-break:break-word;max-width:70ch}
.fig .src::before{content:"SOURCE  ";color:var(--accent);font-weight:500}
.fig .chips{display:flex;gap:6px;flex-wrap:wrap;flex-shrink:0}
.fig .insight{margin:10px 0 0;font-size:14.5px;color:var(--ink);max-width:72ch}

footer{padding:56px 0 72px;color:var(--ink-faint);font-size:12.5px;border-top:1px solid var(--line);margin-top:44px}
footer .mono{color:var(--ink-soft)}
a.txt{color:var(--accent);text-decoration:none}
a.txt:hover{text-decoration:underline}
@media(prefers-reduced-motion:no-preference){html{scroll-behavior:smooth}}
</style>
""")

# nav
navlinks = "".join(
    f'<a href="#{s["id"]}"><span class="n">{s["n"]}</span>&nbsp;{html.escape(s["title"])}</a>'
    for s in SECTIONS)
parts.append(f'<nav><div class="inner"><span class="brand">Data Atlas</span>{navlinks}</div></nav>')

# header
parts.append(f"""<div class="wrap"><header>
  <div class="eyebrow">Final-year project &nbsp;·&nbsp; reference</div>
  <h1>Volleyball Tactical Analysis from Broadcast Video</h1>
  <p class="dek">Twenty-seven views of every dataset, metric and training log in the repository —
  built straight from the raw CSVs and JSON, and grouped by the reviewer each one is for.
  The task: label a one-second rally window by the collective tactic of six players,
  from a single uncalibrated camera.</p>
  <div class="stats">
    <div class="stat"><div class="v">3</div><div class="k">source broadcast videos</div></div>
    <div class="stat"><div class="v">375</div><div class="k">labelled 1&#8209;second clips (v3)</div></div>
    <div class="stat"><div class="v">132</div><div class="k">features per clip</div></div>
    <div class="stat"><div class="v">80,330</div><div class="k">frames in the full-match trace</div></div>
  </div>
  <div class="legend">
    <span style="align-self:center;color:var(--ink-faint);font-family:'IBM Plex Mono',monospace;font-size:11px">READS FOR</span>
    {chip('cv')}{chip('vb')}{chip('ml')}{chip('dl')}
  </div>
</header>""")

# sources table
rows = "".join(
    f'<tr><td class="f">{html.escape(f)}</td><td class="r">{html.escape(r)}</td><td class="d">{html.escape(d)}</td></tr>'
    for f, r, d in SOURCE_ROWS)
parts.append(f"""<div class="sources">
  <h3>Raw artefacts behind the figures</h3>
  <div class="scroll"><table><tbody>{rows}</tbody></table></div>
</div>""")

# sections
for s in SECTIONS:
    figs_html = []
    for path, auds, src, insight in s["figs"]:
        cs = "".join(chip(a) for a in auds)
        figs_html.append(f"""<figure class="fig">
  <div class="imgwrap"><img loading="lazy" alt="{html.escape(insight[:120])}" src="data:image/png;base64,{b64(path)}"></div>
  <figcaption>
    <div class="caphead">
      <div class="src">{html.escape(src)}</div>
      <div class="chips">{cs}</div>
    </div>
    <p class="insight">{html.escape(insight)}</p>
  </figcaption>
</figure>""")
    parts.append(f"""<section id="{s['id']}">
  <div class="s-head"><span class="s-num">{s['n']}</span><h2>{html.escape(s['title'])}</h2></div>
  <p class="s-intro">{html.escape(s['intro'])}</p>
  {''.join(figs_html)}
</section>""")

parts.append("""<footer>
  <p class="mono">Volleyball Tactical Data Atlas &nbsp;·&nbsp; generated from finalyearproject/figures_atlas
  &nbsp;·&nbsp; regenerate with <span style="color:var(--accent)">python make_data_atlas.py</span></p>
  <p>Corpus versions are not interchangeable: v1 (285 clips, pre-detector-fix), v3 (375, fixed pipeline,
  3 videos), v4 (945, 6+ videos). Every figure names the corpus it was computed on. Leave-one-video-out
  is the honest cross-video number throughout; k-fold is reported only to measure the gap.</p>
</footer></div>""")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(parts))
print("wrote", OUT, os.path.getsize(OUT) // 1024, "KB")
