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

    escutar = sub.add_parser("escutar", help="detecta a palavra-chave, grava o comando e confere a voz")
    escutar.add_argument("--salvar-comando", type=Path, default=None, metavar="ARQUIVO.wav",
                         help="salva o último comando gravado (para conferir o corte do VAD)")

    transcrever = sub.add_parser("transcrever", help="transcreve um WAV ou o que você falar depois do bip")
    transcrever.add_argument("arquivo", type=Path, nargs="?", default=None, help="WAV de 16 kHz (opcional)")

    cadastro = sub.add_parser("cadastrar-voz", help="grava frases suas e salva a impressão de voz")
    cadastro.add_argument("--frases", type=int, default=15, help="quantas frases ler (10 a 20)")
    cadastro.add_argument("--segundos", type=float, default=5.0, help="duração máxima de cada frase")

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
    if args.comando == "escutar":
        from chubasa.speaker import criar_verificador
        from chubasa.vad import criar_gravador
        from chubasa.wakeword import comando_escutar

        return comando_escutar(config, dispositivo, criar_verificador(config), criar_gravador(config),
                               args.salvar_comando)
    if args.comando == "transcrever":
        from chubasa.stt import comando_transcrever

        return comando_transcrever(config, dispositivo, args.arquivo)
    if args.comando == "cadastrar-voz":
        from chubasa.speaker import comando_cadastrar_voz

        return comando_cadastrar_voz(config, dispositivo, args.frases, args.segundos)

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
    from chubasa.speaker import ErroVoz
    from chubasa.stt import ErroSTT
    from chubasa.vad import ErroVAD
    from chubasa.wakeword import ErroWakeWord

    try:
        return executar(args, config)
    except ErroDeAudio as erro:
        print(f"Erro de áudio: {erro}", file=sys.stderr)
        return 1
    except ErroWakeWord as erro:
        print(f"Erro na palavra-chave: {erro}", file=sys.stderr)
        return 1
    except ErroVoz as erro:
        print(f"Erro na verificação de voz: {erro}", file=sys.stderr)
        return 1
    except ErroVAD as erro:
        print(f"Erro no detector de fala: {erro}", file=sys.stderr)
        return 1
    except ErroSTT as erro:
        print(f"Erro na transcrição: {erro}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
