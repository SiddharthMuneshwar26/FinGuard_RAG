"""Read-only Google Drive / Docs search (Google Workspace) as an ADK tool.

OAuth 2.0 with exactly one scope: https://www.googleapis.com/auth/drive.readonly
Credentials are never hard-coded. They are loaded, in order, from:
  1. FINGUARD_GOOGLE_OAUTH_SECRET   Secret Manager resource name holding the
                                    authorized-user token JSON, e.g.
                                    projects/<p>/secrets/finguard-drive-token/versions/latest
  2. FINGUARD_GOOGLE_OAUTH_TOKEN_JSON  the token JSON itself (e.g. injected by Cloud Run)
  3. FINGUARD_GOOGLE_OAUTH_TOKEN_FILE  path to a local token file (gitignored)

Create the token once with:
  python -m agents.finguard_agent.workspace_auth --client-secrets client_secret.json --out .secrets/drive_token.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Optional

from google.adk.tools.tool_context import ToolContext

from .turn_state import register_citations

DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
SCOPES = [DRIVE_SCOPE]
MAX_RESULTS = 5
SNIPPET_CHARS = 1500
_GOOGLE_DOC_MIME = "application/vnd.google-apps.document"

ServiceFactory = Callable[[], Any]
_service_factory: Optional[ServiceFactory] = None


def _load_token_info() -> dict[str, Any]:
    secret_name = os.getenv("FINGUARD_GOOGLE_OAUTH_SECRET")
    if secret_name:
        from google.cloud import secretmanager

        client = secretmanager.SecretManagerServiceClient()
        payload = client.access_secret_version(name=secret_name).payload.data
        return json.loads(payload.decode("utf-8"))
    inline = os.getenv("FINGUARD_GOOGLE_OAUTH_TOKEN_JSON")
    if inline:
        return json.loads(inline)
    token_file = os.getenv("FINGUARD_GOOGLE_OAUTH_TOKEN_FILE")
    if token_file and Path(token_file).is_file():
        return json.loads(Path(token_file).read_text(encoding="utf-8"))
    raise RuntimeError(
        "Google Drive credentials are not configured. Set FINGUARD_GOOGLE_OAUTH_SECRET, "
        "FINGUARD_GOOGLE_OAUTH_TOKEN_JSON or FINGUARD_GOOGLE_OAUTH_TOKEN_FILE."
    )


def load_credentials() -> Any:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    info = _load_token_info()
    granted = set(info.get("scopes") or SCOPES)
    if granted - set(SCOPES):
        raise RuntimeError(
            f"The Drive token grants extra scopes {sorted(granted - set(SCOPES))}; "
            f"re-create it with only {DRIVE_SCOPE}."
        )
    credentials = Credentials.from_authorized_user_info(info, scopes=SCOPES)
    if not credentials.valid and credentials.refresh_token:
        credentials.refresh(Request())
    return credentials


def _default_service() -> Any:
    credentials = load_credentials()  # fail fast with a clear message before importing the client
    from googleapiclient.discovery import build

    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def set_service_factory(factory: Optional[ServiceFactory]) -> None:
    """Override the Drive client factory (used by tests)."""
    global _service_factory
    _service_factory = factory


def _service() -> Any:
    return (_service_factory or _default_service)()


def _escape(query: str) -> str:
    return query.replace("\\", "\\\\").replace("'", "\\'")


def search_google_drive(query: str, tool_context: ToolContext, max_results: int = 3) -> dict[str, Any]:
    """Search the user's Google Drive (read-only) for files matching the query.

    Use this only when the user asks about documents in their own Google Drive or Docs.
    Google Docs results include a plain-text snippet. Cite a file as [drive:<file_id>].

    Args:
        query: Words to search for in file names and contents.
        max_results: Number of files to return (1-5).

    Returns:
        A dict with "status" and "results"; each has file_id, name, mime_type, link,
        modified and (for Google Docs) snippet.
    """
    max_results = max(1, min(int(max_results), MAX_RESULTS))
    try:
        service = _service()
        response = service.files().list(
            q=f"fullText contains '{_escape(query.strip())}' and trashed = false",
            pageSize=max_results,
            fields="files(id, name, mimeType, webViewLink, modifiedTime)",
            orderBy="modifiedTime desc",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
    except RuntimeError as exc:
        return {"status": "error", "error_message": str(exc), "results": []}
    except Exception as exc:
        return {"status": "error", "error_message": f"Drive search failed: {exc}", "results": []}

    results: list[dict[str, Any]] = []
    for item in response.get("files", [])[:max_results]:
        result = {
            "file_id": item["id"],
            "name": item.get("name", ""),
            "mime_type": item.get("mimeType", ""),
            "link": item.get("webViewLink", ""),
            "modified": item.get("modifiedTime", ""),
        }
        if result["mime_type"] == _GOOGLE_DOC_MIME:
            try:
                raw = service.files().export(fileId=item["id"], mimeType="text/plain").execute()
                text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
                result["snippet"] = text[:SNIPPET_CHARS]
            except Exception:
                result["snippet"] = ""
        results.append(result)

    register_citations(tool_context, [f"drive:{r['file_id']}" for r in results])
    return {"status": "success" if results else "no_results", "results": results}
