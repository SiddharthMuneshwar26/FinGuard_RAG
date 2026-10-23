"""One-time OAuth 2.0 consent for the Drive tool (drive.readonly scope only).

Usage (PowerShell, from the repo root, agent venv active):
  python -m agents.finguard_agent.workspace_auth --client-secrets .secrets\\client_secret.json --out .secrets\\drive_token.json

`client_secret.json` is a "Desktop app" OAuth client downloaded from
Google Cloud Console > APIs & Services > Credentials. Both files are gitignored
(.secrets/). Upload the token to Secret Manager for cloud use:
  gcloud secrets create finguard-drive-token --data-file=.secrets\\drive_token.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .workspace_tool import SCOPES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--client-secrets", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    from google_auth_oauthlib.flow import InstalledAppFlow

    flow = InstalledAppFlow.from_client_secrets_file(str(args.client_secrets), scopes=SCOPES)
    credentials = flow.run_local_server(port=0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(credentials.to_json(), encoding="utf-8")
    print(f"Saved a drive.readonly token to {args.out}. Keep it out of git.")


if __name__ == "__main__":
    main()
