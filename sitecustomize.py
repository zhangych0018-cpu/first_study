"""本模块在 Python 解释器启动时自动注入项目级运行配置，用于稳定 pytest 和本地脚本的执行环境。它的职责是降低环境差异带来的干扰，而不是承载任何业务逻辑。"""

from __future__ import annotations

import os

os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
