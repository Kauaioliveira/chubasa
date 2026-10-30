"""Entrada da linha de comando: `python -m chubasa [comando]`."""

from __future__ import annotations

import argparse
import sys

from chubasa.config import ConfiguracaoInvalida, iniciar


def criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m chubasa",
        description="Assistente de voz Chubasa. Sem comando, só confere a configuração.",
    )
    parser.add_subparsers(dest="comando", metavar="comando")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = criar_parser().parse_args(argv)
    try:
        iniciar()
    except ConfiguracaoInvalida as erro:
        print(erro, file=sys.stderr)
        return 2

    if args.comando is None:
        print("configuração ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
