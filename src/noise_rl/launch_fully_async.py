"""Launch the non-colocated Slime fully-async rollout path."""

from .launch import main

if __name__ == "__main__":
    main(fully_async=True)
