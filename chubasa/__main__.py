"""Entrada da linha de comando: `python -m chubasa [comando]`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from chubasa.config import Config, ConfiguracaoInvalida, iniciar


def criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m chubasa",
        description="Assistente de voz Chubasa. Sem comando, só confere a configuração.",
    )
    sub = parser.add_subparsers(dest="comando", metavar="comando")

    sub.add_parser("dispositivos", help="lista os microfones disponíveis")

    gravar = sub.add_parser("gravar-teste", help="grava alguns segundos do microfone num WAV")
    gravar.add_argument("--segundos", type=float, default=5.0)
    gravar.add_argument("--saida", type=Path, default=Path("teste.wav"))

    medidor = sub.add_parser("medidor", help="mostra o volume do microfone ao vivo")
    medidor.add_argument("--segundos", type=float, default=None, help="para sozinho depois de N s")

    return parser


def executar(args: argparse.Namespace, config: Config) -> int:
    # Importa só o que o comando usa: mantém `python -m chubasa` rápido.
    from chubasa import audio

    if args.comando == "dispositivos":
        return audio.comando_dispositivos()

    dispositivo = audio.resolver_dispositivo(config.mic_device)
    if args.comando == "gravar-teste":
        return audio.comando_gravar_teste(dispositivo, args.saida, args.segundos)
    if args.comando == "medidor":
        return audio.comando_medidor(dispositivo, args.segundos)

    raise AssertionError(f"comando sem tratamento: {args.comando}")


def main(argv: list[str] | None = None) -> int:
    args = criar_parser().parse_args(argv)
    try:
        config = iniciar()
    except ConfiguracaoInvalida as erro:
        print(erro, file=sys.stderr)
        return 2

    if args.comando is None:
        print("configuração ok")
        return 0

    from chubasa.audio import ErroDeAudio

    try:
        return executar(args, config)
    except ErroDeAudio as erro:
        print(f"Erro de áudio: {erro}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
