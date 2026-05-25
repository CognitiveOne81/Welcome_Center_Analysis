# GitHub Code Display + Email Sender

This repository includes utility scripts for feedback analysis and notifications.

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
python scripts/housing_feedback_clustering.py --clusters 3
```

Optional arguments:
- `--dataset <path>`: custom CSV path.
- `--comments-column <name>`: text column to cluster (default `Comments`).
- `--top-terms <n>`: number of representative terms per cluster.
