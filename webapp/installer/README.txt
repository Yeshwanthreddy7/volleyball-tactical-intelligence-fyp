VOLLEYBALL TACTICAL ANALYSIS  -  Desktop app
Group 56 Final Year Project
==========================================================================

WHAT IT DOES
  Upload a volleyball match video (up to 500 MB). The app tracks the players
  and the ball and gives you back:
    * an annotated MP4 video (Ultra / 720p / 480p / 360p, plays in any player)
    * a TEAM TACTICAL REPORT for the team on the near side (bottom of the
      picture): Coordinated Attack, Coordinated Defense, Delayed Support and
      Spacing Breakdown, with start / end times and confidence
    * an OPPONENT REPORT: every opponent SERVE, SPIKE and BLOCK with its time
  Everything runs on THIS computer. Nothing is uploaded to the internet.

BEFORE YOU INSTALL
  * Windows 10 or 11, 64-bit
  * Internet connection during installation: the installer downloads Python
    and the AI libraries (about 3 GB with an NVIDIA GPU, about 1 GB without).
    A console window shows the progress - keep it open (10-40 minutes).
  * About 6 GB of free disk space
  * 8 GB RAM or more
  * NVIDIA graphics card strongly recommended. Without one the analysis runs
    on the CPU and is very slow: use "Fast" mode and clips under 1 minute.
  * No administrator rights needed.

HOW TO USE
  1. Open "Volleyball Tactical Analysis" from the Desktop or Start menu.
     The first start takes up to a minute.
  2. Analyze video -> drop your video -> optionally choose the part of the
     video -> Upload and analyse.
  3. Wait for the progress bar (about 7 minutes per minute of video on a
     laptop GPU). You can close the window; the analysis finishes in the
     background and appears under "My analyses".
  4. Watch the annotated video, change quality with the gear button, read
     the reports (click a time to jump there) and download MP4 / TXT / CSV.

  Best results: broadcast footage filmed from behind one baseline.
  Your team = the team at the bottom of the picture. Teams change ends
  between sets, so analyse the sets where your team plays at the bottom.

IF SOMETHING GOES WRONG
  * Install failed or app does not open: Start menu ->
    "Repair Volleyball Tactical Analysis" (needs internet).
  * Install log: %LOCALAPPDATA%\VolleyballTacticalAnalysis\install.log
  * Your videos and results: %USERPROFILE%\vta_storage (kept 14 days)
  * Uninstall: Settings -> Apps -> Volleyball Tactical Analysis

Source code: https://github.com/Yeshwanthreddy7/volleyball-tactical-intelligence-fyp
