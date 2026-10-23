"""Make the existing `src` package importable when the agent is run by `adk`.

`adk web agents` and `adk run agents/finguard_agent` put the *agents* folder on
sys.path, not the repository root, so `import src.retrieval` would fail.
For Agent Engine deployments, the deploy script copies `src/` inside this
package folder instead, so the package folder itself is also tried.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parents[1]

for candidate in (_REPO_ROOT, _PACKAGE_DIR):
    if (candidate / "src" / "__init__.py").is_file() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
        break
