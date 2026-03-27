from __future__ import annotations

import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from research_features import load_and_patch_base


def main() -> None:
    base = load_and_patch_base()
    base.main()


if __name__ == "__main__":
    main()
