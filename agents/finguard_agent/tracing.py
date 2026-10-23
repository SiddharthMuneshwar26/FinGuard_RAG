"""Optional export of ADK traces/logs to Cloud Trace and Cloud Logging.

Two ways to turn it on:
  * CLI:  adk web --otel_to_cloud agents      (ADK's built-in flag)
  * Env:  FINGUARD_CLOUD_TRACE=1             (for programmatic runs, e.g. run_agent.py)

Both need Application Default Credentials (`gcloud auth application-default login`),
GOOGLE_CLOUD_PROJECT, the Cloud Trace + Cloud Logging APIs enabled, and the
`google-adk[gcp]` extra (included in requirements-agent.txt).
"""

from __future__ import annotations

import logging

from .settings import ENABLE_CLOUD_TRACE

logger = logging.getLogger(__name__)
_enabled = False


def maybe_enable_cloud_tracing(force: bool = False) -> bool:
    """Wire ADK's OpenTelemetry spans to Google Cloud if the env flag is set."""
    global _enabled
    if _enabled or not (force or ENABLE_CLOUD_TRACE):
        return _enabled
    try:
        import google.auth
        from google.adk.telemetry.google_cloud import get_gcp_exporters, get_gcp_resource
        from google.adk.telemetry.setup import maybe_set_otel_providers

        credentials, project_id = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        hooks = get_gcp_exporters(
            enable_cloud_tracing=True,
            enable_cloud_metrics=False,
            enable_cloud_logging=True,
            google_auth=(credentials, project_id),
        )
        maybe_set_otel_providers(otel_hooks_to_setup=[hooks], otel_resource=get_gcp_resource(project_id))
        _enabled = True
        logger.info("FinGuard: Cloud Trace / Cloud Logging export enabled for project %s", project_id)
    except Exception as exc:  # tracing must never break the agent
        logger.warning("FinGuard: could not enable Cloud Trace (%s); continuing without it", exc)
    return _enabled
