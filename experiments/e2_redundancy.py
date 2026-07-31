"""E2: exact and near-duplicate sensitivity."""

if __package__:
    from ._entrypoint import run
else:
    from _entrypoint import run

if __name__ == "__main__":
    raise SystemExit(run("E2"))
