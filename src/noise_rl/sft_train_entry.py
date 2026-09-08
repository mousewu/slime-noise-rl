"""Enter Slime's offline SFT driver through the shared Ray/SwanLab bridge."""

from .train_entry import main

if __name__ == "__main__":
    main("train_async.py")
