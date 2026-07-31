"""E6: isolated optional adaptive-q integration slot."""

if __package__:
    from ._entrypoint import run
else:
    from _entrypoint import run

if __name__ == "__main__":
    raise SystemExit(run("E6"))
