# General Tour Analysis

## Instant clustering results on every PR

This repo is now configured with GitHub Actions to run a clustering workflow automatically on each pull request.

### What happens when you open/update a PR

1. GitHub Action installs Python dependencies.
2. It runs `scripts/run_clustering.py` on `7-1-2024-5-19-2026 General Feedback.csv`.
3. It uploads `clustering_report.json` as a workflow artifact.
4. It posts the JSON report as a comment directly on the pull request.

### How to use it

1. Push your branch to GitHub.
2. Open a pull request.
3. Go to the **Actions** tab and open the latest **Run clustering on pull requests** run.
4. Read the PR comment or download the artifact.

If you change data format/column names, update `scripts/run_clustering.py`.
