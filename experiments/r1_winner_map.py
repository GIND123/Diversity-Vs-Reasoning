"""R1: headline winner-map factorial."""

if __package__:
    from ._entrypoint import run
else:
    from _entrypoint import run

if __name__ == "__main__":
    raise SystemExit(run("R1"))
