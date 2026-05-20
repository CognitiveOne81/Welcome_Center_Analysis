# Welcome Center Analysis

Qualitative clustering analysis for Welcome Center visitor feedback.

## What the model does

The analysis reads `7-1-2024-5-19-2026 General Feedback.csv`, cleans the `Comments`
field, removes placeholder comments such as `N/A`, converts comments into TF-IDF
text features, and uses KMeans clustering to group similar feedback themes.

The workflow intentionally excludes `Phone` and `Email` from generated output files.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/qualitative_clustering.py
```

Generated files are written to `outputs/`:

- `clustered_feedback.csv`
- `cluster_summary.csv`
- `cluster_terms.csv`
- `cluster_plot.png`
- `metrics.json`

## Pull request automation

Every pull request runs `.github/workflows/clustering.yml`. The workflow installs
Python dependencies, runs the tests, runs the clustering model against the CSV in
the repository, and uploads the generated outputs as a GitHub Actions artifact
named `qualitative-clustering-results`.
