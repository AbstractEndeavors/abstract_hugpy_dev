"""`python -m abstract_hugpy_dev.chaos` -> the runner CLI."""
import sys

from .runner import main

if __name__ == "__main__":
    sys.exit(main())
