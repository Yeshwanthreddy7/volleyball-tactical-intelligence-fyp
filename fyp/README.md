# `fyp/` module docs

See the repo-root [`README.md`](../README.md) — this is the current,
maintained description of the pipeline (dual YOLOv11 detector, BoT-SORT
tracking, court homography, scikit-learn tactical classifier with Mamba
SSM as a comparison arm, 191-test suite, `reproduce.py` harness).

This file used to duplicate that content and drifted out of sync with it
(wrong detector version, wrong tracker, described Mamba as the only model
rather than the comparison arm). Kept as a stub instead of a second copy
so it cannot drift again — edit the root `README.md`, not this file.

Module-level API notes belong in each module's own docstring
(`fyp/pipeline.py --help`, `fyp/AGENT.md` for the reviewer-facing summary).
