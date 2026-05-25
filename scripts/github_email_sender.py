#!/usr/bin/env python3
"""Display code content and send it via email."""

from __future__ import annotations

import argparse
import smtplib
from email.message import EmailMessage
from pathlib import Path


def display_github_code_and_send_email(
    file_path: str,
    recipient_email: str,
    sender_email: str,
    smtp_server: str,
    smtp_port: int,
    smtp_username: str,
    smtp_password: str,
    subject: str = "Code from GitHub",
    use_tls: bool = True,
) -> None:
    """Display code from a file and send it to an email address."""
    code = Path(file_path).read_text(encoding="utf-8")

    print("===== CODE START =====")
    print(code)
    print("===== CODE END =====")

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender_email
    message["To"] = recipient_email
    message.set_content(
        "The requested code is included below.\n\n"
        f"File: {file_path}\n\n"
        f"{code}"
    )

    with smtplib.SMTP(smtp_server, smtp_port, timeout=30) as smtp:
        if use_tls:
            smtp.starttls()
        smtp.login(smtp_username, smtp_password)
        smtp.send_message(message)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", required=True, help="Path to code file")
    parser.add_argument("--to", required=True, dest="recipient_email", help="Recipient email")
    parser.add_argument("--from", required=True, dest="sender_email", help="Sender email")
    parser.add_argument("--smtp-server", required=True, help="SMTP host")
    parser.add_argument("--smtp-port", required=True, type=int, help="SMTP port")
    parser.add_argument("--smtp-username", required=True, help="SMTP username")
    parser.add_argument("--smtp-password", required=True, help="SMTP password")
    parser.add_argument("--subject", default="Code from GitHub", help="Email subject")
    parser.add_argument(
        "--use-tls",
        dest="use_tls",
        action="store_true",
        default=True,
        help="Enable STARTTLS (default: enabled)",
    )
    parser.add_argument(
        "--no-tls",
        dest="use_tls",
        action="store_false",
        help="Disable STARTTLS",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    display_github_code_and_send_email(
        file_path=args.file,
        recipient_email=args.recipient_email,
        sender_email=args.sender_email,
        smtp_server=args.smtp_server,
        smtp_port=args.smtp_port,
        smtp_username=args.smtp_username,
        smtp_password=args.smtp_password,
        subject=args.subject,
        use_tls=args.use_tls,
    )
