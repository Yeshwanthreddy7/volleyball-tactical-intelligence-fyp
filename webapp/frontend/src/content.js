// Static portal content. Edit TEAM below with your real names / roles.

export const TACTICS = [
  { name: "Coordinated Attack", color: "var(--attack)", hex: "#ff8200",
    text: "Fast approach by front-court attackers with base players held near the net." },
  { name: "Coordinated Defense", color: "var(--defense)", hex: "#14af78",
    text: "The team shifts as a rigid unit." },
  { name: "Delayed Support", color: "var(--delayed)", hex: "#e150a0",
    text: "The nearest player reacts late after ball impact." },
  { name: "Spacing Breakdown", color: "var(--spacing)", hex: "#8c5adc",
    text: "Structural failure: the formation is too spread or too clustered." },
];
export const TACTIC_HEX = Object.fromEntries(TACTICS.map((t) => [t.name, t.hex]));

export const OPP_HEX = { SERVE: "#f5b020", SPIKE: "#ef4444", BLOCK: "#38bdf8" };

export const ACTION_HEX = {
  SERVE: "#fAAA1E", RECEPTION: "#3cb45a", PASS: "#3cb45a", DIG: "#28a0a0",
  SET: "#dc3232", SPIKE: "#c81e78", BLOCK: "#7846c8", "FREE BALL": "#6e6e6e",
};

export const OBJECTIVES = [
  { icon: "🎯", title: "Detect and track every player and the ball",
    text: "YOLO11 pose for players, a volleyball fine-tuned YOLO model for the ball, BoT-SORT tracking, all at 1280 px so a 15 px ball is not lost." },
  { icon: "📐", title: "Put the play on a real court",
    text: "Find the court lines in an empty-court base frame automatically and map every player into court metres with a homography." },
  { icon: "🏐", title: "Time-stamp the opponent's key actions",
    text: "Every opponent SERVE, SPIKE and BLOCK with its exact time, from ball-touch events, volleyball touch order and pose." },
  { icon: "🧠", title: "Classify whole-team tactics",
    text: "Coordinated Attack, Coordinated Defense, Delayed Support and Spacing Breakdown from 29-frame windows of the team formation." },
  { icon: "🎬", title: "Deliver a coach-ready output",
    text: "An annotated MP4 that plays everywhere, plus a timestamped team tactical report and a separate opponent action report." },
  { icon: "🔬", title: "Measure it honestly",
    text: "12 classical and 4 deep models compared with leave-one-video-out validation, bootstrap intervals and a majority baseline, reproducible with one command." },
];

export const PROBLEM = {
  short: "Coaches need fast, consistent feedback on how a team coordinates. Manual notation is slow and subjective, and existing sports analytics focus on individual actions or the ball path, not on whole-team spatial coordination.",
  gap: "This project classifies each rally window by the collective tactical pattern of six players: how the formation is shaped, how it moves and whether it moves together, from a single uncalibrated broadcast camera.",
  challenges: [
    "A volleyball is only 15-20 px wide in 720p broadcast video and is often motion-blurred.",
    "Both teams stand within a metre of each other at the net, so team membership cannot come from position alone.",
    "Broadcast video has replays, camera cuts and close-ups that break the court model.",
    "Tracker ids switch during occlusions, so per-player features must survive identity swaps.",
    "Only one camera and no calibration data: every video has a different scale.",
  ],
};

export const PIPELINE = [
  { title: "Upload and probe", code: "webapp/backend/app/main.py",
    text: "The video (up to 500 MB) is uploaded in chunks, checked, and queued for the GPU worker." },
  { title: "Court calibration", code: "BASEFRAMES/extract_base_frames.py, detect_court_corners.py",
    text: "A median 'empty court' base frame is built from the dominant camera shot; the court lines (end, attack and centre lines) are found and fitted to a homography." },
  { title: "Player and ball detection", code: "OUTPUTS/bicausal_extract.py",
    text: "YOLO11m-pose with BoT-SORT tracks every person with 17 keypoints; the project's volleyball fine-tune (volleyball_best.pt) finds the ball at 1280 px." },
  { title: "Teams and players on court", code: "OUTPUTS/bicausal_render.py: build_players",
    text: "Feet are projected into court metres; tracks are split into the near and far team by court side, referees and bench are dropped, six players per side are kept." },
  { title: "Ball track and touches", code: "build_ball, find_touches",
    text: "Static spare balls are suppressed, detections are linked into tracklets, and sharp changes of direction next to a player become touch events." },
  { title: "Action labelling", code: "label_touches",
    text: "Volleyball touch order plus pose (contact above the head) turns touches into SERVE, RECEPTION, SET, SPIKE, BLOCK, DIG, PASS and FREE BALL with exact timestamps." },
  { title: "Team tactics (your team)", code: "NEW_OUTPUTS/code/team_tactics.py, tactical_ml_v6.joblib",
    text: "29-sample windows (about 1 s, resampled for 50 fps video) of the near team's six court positions slide every 1/3 s; the deployed HistGradientBoosting model gives class probabilities, balanced for class frequency, smoothed, with a 0.8 s minimum per call, only during live play." },
  { title: "Opponent actions", code: "bicausal_render.label_touches",
    text: "Touch order and pose turn ball touches into actions. For the far (opponent) team only SERVE, SPIKE and BLOCK are kept, each with its timestamp; all other opponent touches are ignored." },
  { title: "Team tactics rendering", code: "NEW_OUTPUTS/code/new_render.py",
    text: "Your team gets a relation graph and a hull in the tactic colour; opponents are drawn as nodes only. A side panel shows the class probabilities, the tactic timeline and the last opponent serves / spikes / blocks." },
  { title: "Encoding and reports", code: "app/engine.py, app/reports.py",
    text: "H.264 MP4 with the original audio and fast start, plus 720p / 480p / 360p versions, and two timestamped reports: team tactics and opponent actions (TXT, CSV, JSON)." },
];

export const MILESTONES = [
  { date: "Phase 1", title: "Problem definition and data collection",
    text: "11 broadcast match videos collected. Tactical classes and labelling protocol defined.", state: "done" },
  { date: "Phase 2", title: "Detection and tracking baseline",
    text: "Dual detector (stock YOLO for players, volleyball fine-tune for the ball), court mask, ByteTrack / BoT-SORT, jersey-colour team split.", state: "done" },
  { date: "Sep 6, 2026", title: "Audit, fixes and feature redesign",
    text: "Audit of the pipeline; identity-stable slots, NaN-safe features, scale-invariant perm_invariant_v2 descriptor; 191-test suite and reproduce.py.", state: "done" },
  { date: "Sep 29, 2026", title: "13-model comparison",
    text: "9 classical + 4 deep temporal models (InceptionTime, TCN, BiLSTM, Mamba) trained and compared with leave-one-video-out validation; binary attack / defense variant.", state: "done" },
  { date: "Oct 3, 2026", title: "Court calibration and Bi-Causal view",
    text: "Automatic court-line detection on base frames, ball re-detection, rule-based action labelling and Bi-Causal (relation + interaction) annotated videos.", state: "done" },
  { date: "Oct 21, 2026", title: "v6 retrain on 948 clips",
    text: "Full re-extraction and retraining on 948 labelled clips; HistGradientBoosting deployed (LOVO accuracy 67%, macro F1 0.51).", state: "done" },
  { date: "Oct 24, 2026", title: "Web application and documentation portal",
    text: "Full-stack website: upload up to 500 MB, GPU analysis queue, multi-quality annotated MP4, timestamped tactical and opponent reports, MongoDB, desktop app, Windows installer and Vercel deployment.", state: "done" },
  { date: "Now", title: "More matches and labelled clips",
    text: "Grow Delayed Support and Spacing Breakdown (the two minority classes) and add new courts to improve generalisation.", state: "now" },
];

export const TEAM = {
  group: "Group 56",
  department: "Department of Computer Science and Engineering",
  members: [
    { name: "Keshavramprakash D", id: "CB.SC.U4CSE23425" },
    { name: "K. Surya Sekhar", id: "CB.SC.U4CSE23420" },
    { name: "G. Bharath Chandra", id: "CB.SC.U4CSE23414" },
    { name: "V. Punith Sai", id: "CB.SC.U4CSE23768" },
    { name: "U. Yeshwanth Kumar Reddy", id: "CB.SC.U4CS23453" },
  ],
  guide: { name: "Dr. Divya Singh", role: "Assistant Professor, Department of CSE" },
};
