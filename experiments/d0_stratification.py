"""D0: create hardness, tail-heaviness, and entropy strata."""

if __package__:
    from ._entrypoint import run
else:
    from _entrypoint import run

if __name__ == "__main__":
    raise SystemExit(run("D0"))
