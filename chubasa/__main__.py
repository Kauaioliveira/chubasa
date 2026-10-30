"""Entrada da linha de comando: `python -m chubasa [comando]`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from chubasa.config import ARQUIVO_LOG, Config, ConfiguracaoInvalida, iniciar, pasta_de_logs


def criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m chubasa",
        description="Assistente de voz Chubasa. Sem comando, roda o assistente.",
    )
    parser.add_argument("--bandeja", action="store_true",
                        help="roda com ícone na bandeja, sem janela (use com pythonw)")
    sub = parser.add_subparsers(dest="comando", metavar="comando")

    sub.add_parser("config", help="só confere a configuração (.env) e sai")

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

    perguntar = sub.add_parser("perguntar", help="conversa com o Claude por texto (testa o Bloco 5)")
    perguntar.add_argument("perguntas", nargs="*", help="perguntas em sequência; sem nenhuma, modo interativo")
    perguntar.add_argument("--falar", action="store_true", help="fala as respostas (TTS) frase por frase")

    falar = sub.add_parser("falar", help="fala um texto com a voz configurada")
    falar.add_argument("texto")

    cadastro = sub.add_parser("cadastrar-voz", help="grava frases suas e salva a impressão de voz")
    cadastro.add_argument("--frases", type=int, default=15, help="quantas frases ler (10 a 20)")
    cadastro.add_argument("--segundos", type=float, default=5.0, help="duração máxima de cada frase")

    return parser


def executar(args: argparse.Namespace, config: Config) -> int:
    # Importa só o que o comando usa: mantém os comandos rápidos.
    if args.comando == "perguntar":
        from chubasa.brain import comando_perguntar

        return comando_perguntar(config, args.perguntas, args.falar)
    if args.comando == "falar":
        from chubasa.tts import comando_falar

        return comando_falar(config, args.texto)

    from chubasa import audio

    if args.comando == "dispositivos":
        return audio.comando_dispositivos()

    dispositivo = audio.resolver_dispositivo(config.mic_device)
    if args.bandeja:
        from chubasa.tray import comando_bandeja

        return comando_bandeja(config, dispositivo)
    if args.comando is None:
        from chubasa.loop import comando_assistente

        return comando_assistente(config, dispositivo)
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
    # O assistente (console ou bandeja) também registra em arquivo; a bandeja, só em arquivo.
    roda_assistente = args.bandeja or args.comando is None
    arquivo_log = pasta_de_logs() / ARQUIVO_LOG if roda_assistente else None
    try:
        config = iniciar(arquivo_log=arquivo_log, console=not args.bandeja)
    except ConfiguracaoInvalida as erro:
        print(erro, file=sys.stderr)
        if args.bandeja and arquivo_log is not None:  # pythonw não tem console: deixa no log
            arquivo_log.parent.mkdir(parents=True, exist_ok=True)
            with arquivo_log.open("a", encoding="utf-8") as arquivo:
                arquivo.write(f"{erro}\n")
        return 2

    if args.comando == "config":
        print("configuração ok")
        return 0

    from chubasa.audio import ErroDeAudio
    from chubasa.speaker import ErroVoz
    from chubasa.stt import ErroSTT
    from chubasa.tts import ErroTTS
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
    except ErroTTS as erro:
        print(f"Erro na voz: {erro}", file=sys.stderr)
        return 1


def main_bandeja() -> int:
    """Entrada do chubasa-bandeja (sem console)."""
    return main(["--bandeja"])


if __name__ == "__main__":
    sys.exit(main())
