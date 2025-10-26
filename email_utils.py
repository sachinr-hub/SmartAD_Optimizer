import os
import smtplib
import ssl
from email.message import EmailMessage
from typing import Optional
from dotenv import load_dotenv
import json
from pathlib import Path

try:
    import streamlit as st  # Optional, only for st.secrets fallback
except Exception:
    st = None

# Load environment from .env if present
load_dotenv()

# Environment variables expected:
# - EMAIL_ADDRESS: Your Gmail address (e.g., example@gmail.com)
# - EMAIL_APP_PASSWORD: Gmail App Password (NOT your regular Gmail password)
# - FRONTEND_URL (optional): Base URL of your Streamlit app, e.g., https://your-app.streamlit.app


def _resolve_config() -> tuple[str, str, str]:
    """Resolve (email_address, email_app_password, frontend_url) from multiple sources.

    Priority:
      1) Environment variables EMAIL_ADDRESS, EMAIL_APP_PASSWORD, FRONTEND_URL
      2) streamlit secrets: st.secrets["email"].{address, app_password, frontend_url}
      3) config/system_config.json -> email.{address, app_password, frontend_url}
    """
    # 1) Environment variables
    email_address = (os.getenv("sachinroy2091@gmail.com") or "").strip()
    email_password = (os.getenv("cydlcapeakrnmyxn") or "").strip()
    frontend_url = (os.getenv("http://localhost:8501") or "").strip()

    # 2) Streamlit secrets
    if (not email_address or not email_password) and st is not None:
        try:
            email_secrets = st.secrets.get("email", {})
            if not email_address:
                email_address = str(email_secrets.get("address", "")).strip()
            if not email_password:
                email_password = str(email_secrets.get("app_password", "")).strip()
            if not frontend_url:
                frontend_url = str(email_secrets.get("frontend_url", "")).strip()
        except Exception:
            pass

    # 3) JSON config fallback
    if not (email_address and email_password):
        try:
            config_path = Path(__file__).parent / "config" / "system_config.json"
            if not config_path.exists():
                # project root /config/system_config.json
                config_path = Path(__file__).parent / ".." / "config" / "system_config.json"
                config_path = config_path.resolve()
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            email_cfg = cfg.get("email", {})
            if not email_address:
                email_address = str(email_cfg.get("address", "")).strip()
            if not email_password:
                email_password = str(email_cfg.get("app_password", "")).strip()
            if not frontend_url:
                frontend_url = str(email_cfg.get("frontend_url", "")).strip()
        except Exception:
            pass

    # Normalize password by removing spaces (Gmail may display with spaces)
    email_password = email_password.replace(" ", "")
    return email_address, email_password, frontend_url


def send_password_reset_email(to_email: str, token: str) -> tuple[bool, str]:
    """
    Send a password reset email via Gmail.

    Returns (success, message).
    """
    # Resolve configuration from env, secrets, or config file
    email_address, email_password, frontend_url = _resolve_config()

    if not email_address or not email_password:
        return False, (
            "Email not configured. Please set EMAIL_ADDRESS and EMAIL_APP_PASSWORD environment variables. "
            "For Gmail, create an App Password and use that instead of your normal password."
        )

    # Build reset link if FRONTEND_URL provided, otherwise include token instructions
    if frontend_url and frontend_url.strip():
        if frontend_url.endswith('/'):
            frontend_url = frontend_url[:-1]
        reset_link = f"{frontend_url}?reset_token={token}"
        extra_instructions = ""
    else:
        reset_link = None
        extra_instructions = (
            "\n\nNote: The application FRONTEND_URL is not configured. "
            "Paste the following token into the Reset Password form in the app: "
            f"{token}"
        )

    subject = "Reset your SmartAd Optimizer password"

    body_lines = [
        "Hello,",
        "\nWe received a request to reset your password.",
    ]
    if reset_link:
        body_lines.append("\nClick the link below to reset your password:")
        body_lines.append(reset_link)
    else:
        body_lines.append("\nA direct reset link couldn't be generated because FRONTEND_URL is not set.")
        body_lines.append("Use the token below in the app's Reset Password form:")
        body_lines.append(token)

    body_lines.append("\nThis link or token will expire in 60 minutes.")
    body_lines.append("If you did not request a password reset, you can safely ignore this email.")
    if extra_instructions:
        body_lines.append(extra_instructions)

    body = "\n".join(body_lines)

    msg = EmailMessage()
    msg["From"] = email_address
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            # Optional debug logging if EMAIL_DEBUG=1
            try:
                if os.getenv("EMAIL_DEBUG") == "1":
                    server.set_debuglevel(1)
            except Exception:
                pass
            server.login(email_address, email_password)
            server.send_message(msg)
        return True, "Password reset email sent."
    except Exception as e:
        hint = ""
        emsg = str(e)
        if "535" in emsg or "Username and Password not accepted" in emsg:
            hint = (
                " Check EMAIL_APP_PASSWORD (must be a 16-character Gmail App Password without spaces) "
                "and ensure 2-Step Verification is enabled."
            )
        elif "[Errno" in emsg or "timed out" in emsg:
            hint = " Check your network/Firewall for smtp.gmail.com:465 access."
        return False, f"Failed to send email: {emsg}.{hint}"
