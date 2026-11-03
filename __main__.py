"""Точка входа для `python -m xanosmerks.meanflare`."""
import sys
from .engine import main

if __name__ == "__main__":
    sys.exit(main())