"""R2: conditions under which diversity or coverage hurts."""

if __package__:
    from ._entrypoint import run
else:
    from _entrypoint import run

if __name__ == "__main__":
    raise SystemExit(run("R2"))
