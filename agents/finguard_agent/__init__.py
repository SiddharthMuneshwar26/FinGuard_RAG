"""FinGuard Google ADK agent package.

`adk web agents` / `adk run agents/finguard_agent` discover the agent through
`root_agent`, defined in `agent.py`.
"""

from . import _bootstrap  # noqa: F401  (puts the repo root on sys.path)
from . import agent  # noqa: F401
