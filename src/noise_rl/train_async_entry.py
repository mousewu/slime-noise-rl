"""Enter Slime's fully-async driver through the project's isolated Ray setup."""

from .train_entry import main

if __name__ == "__main__":
    main("train_async.py")
