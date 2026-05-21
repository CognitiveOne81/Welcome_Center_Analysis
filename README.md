# GitHub Code Display + Email Sender

This repository has been reset to a single-purpose utility.

## What it does

The script in `scripts/github_email_sender.py` contains one function:

- `display_github_code_and_send_email(...)`

That function:
1. Reads a local code file.
2. Prints the code to stdout (for GitHub Actions logs / terminal display).
3. Emails the same code to your recipient address via SMTP.

## Usage

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
