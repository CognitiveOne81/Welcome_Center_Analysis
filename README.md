# GitHub Code Display + Email Sender

This repository includes utility scripts for feedback analysis and notifications.

## Automated GitHub feature (pre-merge run + email on merge)

A GitHub Actions workflow is included at:

- `.github/workflows/model-preview-and-email.yml`

It does two things:

1. **Before merge (Pull Request to `main`)**
   - Installs dependencies.
   - Runs the clustering model in GitHub Actions.
   - Uploads the output as an artifact (`model-output-preview`) so you can review results before clicking merge.

2. **After merge (Push to `main`)**
   - Re-runs the clustering model on the merged code.
   - Emails the run output (`model_output.txt`) using SMTP credentials stored as GitHub repository secrets.

### Required GitHub secrets

Set these in **Settings → Secrets and variables → Actions**:

- `RECIPIENT_EMAIL`
- `SENDER_EMAIL`
- `SMTP_SERVER`
- `SMTP_PORT`
- `SMTP_USERNAME`
- `SMTP_PASSWORD`

## Scripts

### 1) Display GitHub code and send email

The script in `scripts/github_email_sender.py` contains one function:

- `display_github_code_and_send_email(...)`

That function:
1. Reads a local code file.
2. Prints the code to stdout (for GitHub Actions logs / terminal display).
3. Emails the same code to your recipient address via SMTP.

Usage:

```bash
python scripts/github_email_sender.py \
  --file README.md \
  --to you@example.com \
  --from sender@example.com \
  --smtp-server smtp.example.com \
  --smtp-port 587 \
  --smtp-username your_user \
  --smtp-password your_password
```

Optional:
- `--subject "Custom subject"`
- `--use-tls` (recommended for port 587)

### 2) Cluster housing feedback comments

The script `scripts/housing_feedback_clustering.py` builds a text clustering model for the **Housing Feedback** dataset (`7-1-2024 -5-19-2026 Housing Feedback.csv`) using TF-IDF + KMeans.

Install dependencies:

```bash
pip install -r requirements.txt
```

Run clustering:

```bash
python scripts/housing_feedback_clustering.py --high-rating-clusters 3 --low-rating-clusters 3
```

Optional arguments:
- `--dataset <path>`: custom CSV path.
- `--comments-column <name>`: text column to cluster (default `Comments`).
- `--rating-column <name>`: rating column used to split cohorts (default `Rating`).
- `--high-rating-threshold <num>`: split threshold (default `4.0`).
- `--high-rating-clusters <n>`: clusters for ratings `>= threshold` (default `3`).
- `--low-rating-clusters <n>`: clusters for ratings `< threshold` (default `3`).
- `--top-terms <n>`: number of representative terms per cluster.
