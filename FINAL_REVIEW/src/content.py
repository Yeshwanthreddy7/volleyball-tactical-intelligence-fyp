"""All slide / report text in one place (simple wording). Both build scripts read this.

Every number here is traceable:
  model results ......... YESHWANTH_PDF/evidence/final_report_eval*.json (945 clips, 8 videos, LOVO)
  functional tests ...... FINAL_REVIEW/evidence/functional_tests.json, pytest_result.txt
  full analysis run ..... %USERPROFILE%/vta_storage/jobs/e507171f.../result.json
"""

COURSE = "23CSE498 -- Project Phase II (External Review)"
TEAM_NO = "56"
TITLE = "Court-Calibrated Group Activity Recognition for Volleyball Team Behaviour Analysis Using Monocular Video"
SUBTITLE = "A web application that finds team tactics in broadcast volleyball video"
STUDENTS = [
    ("CB.SC.U4CSE23415", "G. Bharath Chandra"),
    ("CB.SC.U4CSE23420", "K. Surya Sekhar"),
    ("CB.SC.U4CSE23425", "Keshavramprakash D."),
    ("CB.SC.U4CSE23453", "U. Yeshwanth Kumar Reddy"),
    ("CB.SC.U4CSE23768", "V. Punith Sai"),
]
GUIDE = ("Dr. Divya Singh", "Assistant Professor")
REPO = "github.com/Yeshwanthreddy7/volleyball-tactical-intelligence-fyp"
SITE = "volleyballanalyzer.dpdns.org"

PROBLEM = [
    ("Real-World Problem", "Coaches watch match videos by hand to see how their team moves together. "
     "It takes hours, and two people can judge the same rally differently."),
    ("Target Users", "Volleyball coaches, analysts and players at clubs and colleges, who usually have only "
     "one camera and no tracking equipment."),
]
FR = [
    ("FR-1", "Upload a match video (MP4, MOV, AVI, MKV, WebM) and choose the part to analyse."),
    ("FR-2", "Find and track players and the ball; give the near-side team one of 4 tactics for every second of play."),
    ("FR-3", "Show an annotated video and give downloadable reports: tactic timeline and opponent SERVE / SPIKE / BLOCK."),
    ("FR-4", "Keep a list of each user's past analyses."),
]
NFR = [
    ("NFR-1", "Refuse wrong or non-volleyball files early with a clear message; show progress while working."),
    ("NFR-2", "Run on a normal laptop with a graphics card (NVIDIA RTX 3050); one camera, no sensors."),
]

ARCH_SUMMARY = ("The coach uploads a video on the website. The server checks it and puts it in a queue. "
                "A worker runs the analysis one job at a time, and the results page shows the video and reports.")
PIPE_POINTS = [
    ("1. Safety check", "is it a real video, with a court and players?"),
    ("2. Detection", "find the players, the ball and the court lines."),
    ("3. Tracking", "give each player the same ID in every frame."),
    ("4. Team + features", "split the two teams; measure spacing, speed and ball position each second."),
    ("5. Tactic + report", "the HGB model names the tactic; we save the video and reports."),
]

COMPONENTS = [
    ("Website (React)", "the pages the user sees: upload, progress, results, my analyses."),
    ("Server (FastAPI)", "receives the video, checks it and keeps the job queue."),
    ("Analysis Engine (Python)", "finds players and ball, names the tactic, makes the output video."),
]
JOURNEY = [
    ("User Journey", "Home, then Analyze (upload, pick time range and mode), then Progress, then Results."),
    ("Results Page", "play the annotated video, see the tactic timeline, download the reports."),
]
DATA = [
    ("Data Storage", "MongoDB keeps one record per analysis; the video and reports are saved in a folder."),
    ("External Parts", "graphics card (GPU) for fast detection; FFmpeg to make the output video."),
]
PLATFORM = [
    ("Tools Used", "Python, FastAPI, React, MongoDB, YOLO11, scikit-learn, OpenCV, FFmpeg."),
    ("Runs On", "Windows laptop with an RTX 3050 graphics card; also as a desktop app."),
]

MODULES = [
    ("Module 1: User Interface", "Home, Analyze, Progress, Results, My analyses, Demos, Methodology, "
     "Documentation and Team pages; desktop app.", "Complete"),
    ("Module 2: Core Analysis", "Video check, player and ball detection, tracking, team split, tactic model, "
     "opponent actions, output video, reports and clips.", "Working; known accuracy limits"),
    ("Module 3: Data & Storage", "A database record and a folder for every analysis; old analyses deleted "
     "after 14 days.", "Complete"),
    ("Module 4: External Integration", "Graphics card for fast detection, FFmpeg for video, public project "
     "website; Azure CPU cloud server.", "Complete (GPU cloud: future)"),
]
ENV = [
    ("Execution Platform", "Windows 11 laptop with an RTX 3050 graphics card. One script, run_website.bat, "
     "starts everything and opens the website."),
    ("Source Repository", REPO + " (version 1.0.0)."),
]

MOD_A = ("Video Upload & Check", [
    ("How it works", "the video is sent in small parts with a progress bar; the server then checks a few "
     "frames for a court with players."),
    ("Input & Output", "video, time range and mode in; a job number out, or a clear error message."),
    ("Checks", "only video files; size limit; must show a volleyball court; only the uploader can cancel or delete."),
])
MOD_B = ("Tactic Analysis Engine", [
    ("How it works", "find players and ball, follow them, split the teams, measure the team every second, "
     "then the model names the tactic. Replays are skipped."),
    ("Connection", "runs separately from the website, so the website keeps working while a video is analysed."),
])
MOD_C = ("Tactic Model", [
    ("What it does", "names the team's tactic every second: Coordinated Attack, Coordinated Defense, "
     "Delayed Support or Spacing Breakdown."),
    ("Why HGB", "we compared 16 models; HGB was among the most accurate and runs fast, even without a "
     "graphics card."),
])
MOD_D = ("Outputs", [
    ("User gets", "annotated video with sound, tactic report, opponent SERVE / SPIKE / BLOCK report, "
     "and short example clips."),
])
MODEL_TABLE = (["Model", "Type", "Correct Tactic"], [
    ["HGB (our model)", "Machine learning", "63.7%"],
    ["InceptionTime", "Deep learning", "61.9%"],
    ["Mamba", "Deep learning", "46.1%"],
    ["Guessing the most common tactic", "Baseline", "39.0%"],
])

TESTS = (["Test Case", "Input / Action", "Result Seen", "Status"], [
    ["TC-01: Full analysis", "150-second match video, Accurate mode",
     "annotated video, 40 tactic segments, 9 opponent actions, clips for all 4 tactics", "Pass"],
    ["TC-02: Real match check", "all 11 match videos in our dataset", "all accepted", "Pass"],
    ["TC-03: Not volleyball", "street photo, test pattern, screen recording", "refused with a clear message", "Pass"],
    ["TC-04: Wrong file type", "a PDF document", "refused; allowed types shown", "Pass"],
    ["TC-05: Fake video file", "a text file renamed to .mp4", "detected and refused", "Pass"],
    ["TC-06: File too large", "a 5 GB file", "refused before upload starts", "Pass"],
    ["TC-07: Security check", "asked for a missing analysis and a hidden server file", "refused; nothing shown", "Pass"],
    ["TC-08: Automated tests", "266 automatic code checks", "all 266 passed", "Pass"],
])
VERIFY = [
    ("Testing", "all 8 test cases passed, and 3 full analyses finished on the laptop."),
    ("Speed & Accuracy", "about 7 minutes per minute of video on the laptop. On videos the model never saw, "
     "it is right 63.7% of the time (macro-F1 0.54), against 39.0% by guessing. Attack and Defense are "
     "reliable (F1 about 0.76); Delayed Support and Spacing Breakdown are not yet (F1 about 0.3)."),
]
DEMO = ("We open the website on the laptop, upload a one-minute match clip, choose Accurate mode and watch the "
        "progress bar. Then we play the annotated video, show the tactic timeline and download the reports.")

JUST = [
    ("1. Why These Tools", [
        "Python is the standard choice for video analysis and machine learning.",
        "FastAPI handles big uploads well; React makes a fast, simple website; MongoDB stores job records easily.",
    ]),
    ("2. Design Trade-offs", [
        "One analysis at a time, so the 4 GB graphics card never runs out of memory.",
        "The analysis runs separately, so a problem in one video does not stop the website.",
    ]),
    ("3. Security & Error Handling", [
        "Only real video files are accepted; only the uploader can cancel or delete a job.",
        "If the page is closed, the job stops after 5 minutes; if court lines are not found, a standard court is used.",
    ]),
    ("4. Limits & Future Scope", [
        "Works after the match, not live; covers the near-side team.",
        "Now: public site on an Azure CPU server (Fast mode). Next: a cloud server with a GPU, more match videos, better ball detection.",
    ]),
]

LIMITS = [
    "Analysis is after the match, not live: about 7 minutes per minute of video.",
    "Tactics are given for the near-side team; teams change sides each set, so we analyse one set at a time.",
    "Delayed Support and Spacing Breakdown are recognised less reliably than attack and defense.",
    "The model learned from 945 clips of 8 match videos labelled by one annotator; more data and a second annotator will help.",
    "Opponent SERVE / SPIKE / BLOCK rules were right in 17 of 31 hand-checked events; missed events are not yet counted.",
]
FUTURE = [
    "Run the website on a cloud server with a graphics card for faster results.",
    "Add more match videos and labels to improve accuracy.",
    "Finish the new ball detector, now being trained on all 11 videos.",
    "Recognise team names and follow the same team across sets.",
]

CERTS = [
    ("G. Bharath Chandra", "Python for Computer Vision with OpenCV and Deep Learning", "Udemy", "10"),
    ("K. Surya Sekhar", "Computer Vision Basics", "Coursera (University at Buffalo)", "12"),
    ("Keshavramprakash D.", "Master Computer Vision: OpenCV4 in Python with Deep Learning", "Coursera", "16"),
    ("U. Yeshwanth Kumar Reddy", "Python for Computer Vision with OpenCV and Deep Learning", "Udemy", "10"),
    ("V. Punith Sai", "Python for Computer Vision with OpenCV and Deep Learning", "Udemy", "10"),
]

REFS = [
    ("zhang2024bicausal", "Y. Zhang, W. Liu, D. Xu, Z. Zhou, and Z. Wang, ``Bi-Causal: Group activity recognition via bidirectional causality,'' in \\emph{Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)}, pp. 1450--1459, 2024."),
    ("ibrahim2016", "M. S. Ibrahim, S. Muralidharan, Z. Deng, A. Vahdat, and G. Mori, ``A hierarchical deep temporal model for group activity recognition,'' in \\emph{Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)}, pp. 1971--1980, 2016."),
    ("yolo11", "G. Jocher and J. Qiu, ``Ultralytics YOLO11.'' https://github.com/ultralytics/ultralytics, 2024."),
    ("botsort", "N. Aharon, R. Orfaig, and B.-Z. Bobrovsky, ``BoT-SORT: Robust associations multi-pedestrian tracking,'' arXiv:2206.14651, 2022."),
    ("sklearn", "F. Pedregosa \\emph{et al.}, ``Scikit-learn: Machine learning in Python,'' \\emph{Journal of Machine Learning Research}, vol. 12, pp. 2825--2830, 2011."),
    ("grinsztajn2022", "L. Grinsztajn, E. Oyallon, and G. Varoquaux, ``Why do tree-based models still outperform deep learning on typical tabular data?,'' in \\emph{Proc. NeurIPS Datasets and Benchmarks Track}, 2022."),
    ("inceptiontime", "H. Ismail Fawaz \\emph{et al.}, ``InceptionTime: Finding AlexNet for time series classification,'' \\emph{Data Mining and Knowledge Discovery}, vol. 34, pp. 1936--1962, 2020."),
    ("hartley2004", "R. Hartley and A. Zisserman, \\emph{Multiple View Geometry in Computer Vision}, 2nd ed. Cambridge University Press, 2004."),
    ("mot2008", "K. Bernardin and R. Stiefelhagen, ``Evaluating multiple object tracking performance: The CLEAR MOT metrics,'' \\emph{EURASIP Journal on Image and Video Processing}, 2008."),
    ("mamba", "A. Gu and T. Dao, ``Mamba: Linear-time sequence modeling with selective state spaces,'' arXiv:2312.00752, 2023."),
    ("fastapi", "S. Ram\\'irez, ``FastAPI documentation.'' https://fastapi.tiangolo.com, 2024."),
    ("react", "Meta Open Source, ``React documentation.'' https://react.dev, 2024."),
    ("mongodb", "MongoDB Inc., ``MongoDB 7.0 manual.'' https://www.mongodb.com/docs/manual, 2024."),
]
