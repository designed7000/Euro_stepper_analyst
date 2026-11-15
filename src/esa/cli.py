"""Command line entry point for the esa platform."""

from importlib.metadata import version as package_version

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def main() -> None:
    """Euro Stepper Analyst data platform."""


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(f"esa {package_version('esa')}")
