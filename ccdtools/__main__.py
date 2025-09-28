"""Run ccdtools as a module. This delegates to the CLI by default."""
from . import cli


if __name__ == "__main__":
    cli.main()
