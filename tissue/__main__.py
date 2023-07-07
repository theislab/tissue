"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """Tissue."""


if __name__ == "__main__":
    main(prog_name="tissue")  # pragma: no cover
