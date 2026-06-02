"""Project-local Python startup customizations.

This repository disables third-party pytest plugin autoload by default because
the current Anaconda environment contains plugins that can block startup for
unrelated reasons.  Built-in pytest plugins still load normally.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
