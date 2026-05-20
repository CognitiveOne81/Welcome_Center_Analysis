# General Tour Analysis

## Instant comment-theme analysis on every PR

This repo is configured with GitHub Actions to run topic/phrase extraction automatically on each pull request.

### Recommended model for your goal (common phrases + ideas)

Use **NMF topic modeling on TF-IDF vectors with n-grams**, then extract top bi/tri-grams per topic.

Why this is a good fit:
- Works well on short feedback comments.
- Produces interpretable topic keywords.
- Captures repeated phrases (e.g., "tour guide", "wait time") via n-grams.
- Runs fast and reliably in CI.

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

If your data format/column names change, update `scripts/run_clustering.py`.
