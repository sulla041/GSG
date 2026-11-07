"""V3.03 纯真有向边退化版本独立启动脚本。"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from code.v3_03.edge_only_training import main

if __name__ == "__main__":
    main()
