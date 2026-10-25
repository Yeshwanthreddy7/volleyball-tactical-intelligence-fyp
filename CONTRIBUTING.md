# Contributing

Thank you for your interest in improving this project. This guide explains how to set up a development copy, the rules the code follows, and how to submit a change.

## Development setup

1. Fork the repository and clone your fork.
2. Install the dependencies by following [docs/INSTALLATION.md](docs/INSTALLATION.md).
3. Check that the test suite passes before you change anything:

   ```bash
   python -m pytest fyp/tests -q
   ```

## Branches and commits

- Create a branch from `main`: `feature/<short-name>`, `fix/<short-name>` or `docs/<short-name>`.
- Write commit messages in the [Conventional Commits](https://www.conventionalcommits.org/) style:

  ```text
  feat(court): detect court corners on light floors
  fix(pipeline): reject ball points outside the frame
  docs: add troubleshooting for CUDA installs
  ```

  Common types: `feat`, `fix`, `docs`, `refactor`, `test`, `build`, `chore`.
- Keep each commit to one logical change.

## Code rules

- **Feature-shaping flags are a contract.** Any change to detection, tracking, team separation or court mapping changes the features. If you change one, re-extract the training CSVs and retrain, or the models will be served data they never saw. See [docs/TRAINING.md](docs/TRAINING.md) §0.
- **Report leave-one-video-out numbers.** Within-video k-fold scores overstate accuracy and are not accepted as headline results.
- **Add a test for every bug fix.** Most tests in `fyp/tests/` replay a real failure; follow that pattern.
- **Keep GPU-free paths GPU-free.** Modules such as `teams.py`, `identity.py`, `detect_utils.py` and `train_utils.py` must import without torch.
- **Do not commit videos, caches or files over 100 MB.** `.gitignore` already excludes `*.mp4`, `dataset/` and the large caches.
- Follow the style of the surrounding code: type hints, docstrings that explain *why*, and comments that cite the audit ID (for example `M11 (AUDIT_FINDINGS.md)`) when a line exists because of a finding.

## Pull requests

1. Run the tests and make sure they pass.
2. Update the documentation if behaviour, flags or outputs change.
3. Add an entry under an `Unreleased` heading in [CHANGELOG.md](CHANGELOG.md).
4. Open the pull request and fill in the template.

## Reporting bugs and requesting features

Use the issue templates. For a bug, include the full command, the error, your OS, `python --version`, and whether a GPU was used.
