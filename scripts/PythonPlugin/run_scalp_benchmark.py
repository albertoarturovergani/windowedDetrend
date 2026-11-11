"""Run the 64-channel benchmark directly from source, without installation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent/'src'))
from windowed_detrend.scalp_benchmark import main

if __name__ == '__main__':
    main()
