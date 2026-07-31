"""R4: tail-heaviness conditioning."""

if __package__:
    from ._entrypoint import run
else:
    from _entrypoint import run

if __name__ == "__main__":
    raise SystemExit(run("R4"))
