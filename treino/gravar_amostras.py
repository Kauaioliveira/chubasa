"""Grava amostras da sua voz para treinar o modelo "hey chubasa".

Uso (com o venv ativo, na raiz do projeto):
    python treino/gravar_amostras.py --positivas 30 --negativas 30

Gera WAVs de 16 kHz, mono, 16 bits em treino/amostras/positivas e treino/amostras/negativas,
com o silêncio das pontas aparado, e no fim empacota tudo em treino/amostras.zip para subir no
Colab. Nada disso vai para o git (.gitignore). Rodar de novo continua a numeração.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np

from chubasa.audio import TAXA, ErroDeAudio, dbfs, gravar, resolver_dispositivo, salvar_wav
from chubasa.config import ConfiguracaoInvalida, carregar_config

PASTA_PADRAO = Path(__file__).resolve().parent / "amostras"

# Frases parecidas com a palavra-chave (o modelo aprende a NÃO disparar com elas) e fala comum.
FRASES_NEGATIVAS = [
    "hey chuva",
    "ei, chuveiro",
    "chupa-cabra",
    "e aí, beleza?",
    "hey Bruna",
    "cuba libre",
    "hey Siri",
    "chama a Samara",
    "que chuva forte hoje",
    "passa o sal, por favor",
    "acabou a bateria",
    "vou sair daqui a pouco",
    "hey, chega aqui",
    "o ônibus já passou",
    "liga a luz da sala",
    "chuchu com abóbora",
    "obrigado, até amanhã",
    "você viu o jogo ontem?",
    "hey Jarvis",
    "bota uma música aí",
]

MINIMO_FALA_S = 0.3
VOLUME_MINIMO_DBFS = -45.0


def aparar_silencio(audio: np.ndarray, taxa: int = TAXA, margem_s: float = 0.15,
                    acima_do_ruido_db: float = 10.0, abaixo_do_pico_db: float = 30.0,
                    piso_dbfs: float = -50.0) -> np.ndarray:
    """Corta o silêncio das pontas, mantendo uma margem. Devolve vazio se não houver fala.

    Fala = trechos de 20 ms pelo menos `acima_do_ruido_db` acima do ruído de fundo (estimado pelo
    percentil 10 da energia). Ruído constante, como ventilador, não conta como fala.
    """
    janela = int(0.02 * taxa)
    n = len(audio) // janela
    if n == 0:
        return audio[:0]
    blocos = audio[: n * janela].astype(np.float64).reshape(n, janela) / 32768.0
    energia_db = 10 * np.log10(np.mean(blocos**2, axis=1) + 1e-12)
    ruido_db = np.percentile(energia_db, 10)
    limiar = max(ruido_db + acima_do_ruido_db, energia_db.max() - abaixo_do_pico_db, piso_dbfs)
    ativos = np.flatnonzero(energia_db >= limiar)
    if ativos.size == 0:
        return audio[:0]
    margem = int(margem_s * taxa)
    inicio = max(0, ativos[0] * janela - margem)
    fim = min(len(audio), (ativos[-1] + 1) * janela + margem)
    return audio[inicio:fim]


def proximo_indice(pasta: Path, prefixo: str) -> int:
    existentes = [int(p.stem.rsplit("_", 1)[-1]) for p in pasta.glob(f"{prefixo}_*.wav")
                  if p.stem.rsplit("_", 1)[-1].isdigit()]
    return max(existentes, default=0) + 1


def problema_na_amostra(aparado: np.ndarray, taxa: int = TAXA) -> str | None:
    if aparado.size < MINIMO_FALA_S * taxa:
        return "não ouvi fala nenhuma"
    if dbfs(aparado) < VOLUME_MINIMO_DBFS:
        return "ficou baixo demais, chegue mais perto do microfone"
    return None


def gravar_serie(pasta: Path, prefixo: str, quantidade: int, segundos: float, dispositivo: int | None,
                 instrucao) -> int:
    pasta.mkdir(parents=True, exist_ok=True)
    inicio = proximo_indice(pasta, prefixo)
    gravadas = 0
    while gravadas < quantidade:
        numero = inicio + gravadas
        texto = instrucao(gravadas)
        input(f"\n[{gravadas + 1}/{quantidade}] Enter e diga: \"{texto}\"")
        print("  gravando...")
        aparado = aparar_silencio(gravar(segundos, dispositivo))
        problema = problema_na_amostra(aparado)
        if problema:
            print(f"  {problema}. Vamos de novo.")
            continue
        destino = pasta / f"{prefixo}_{numero:03d}.wav"
        salvar_wav(destino, aparado)
        print(f"  ok ({len(aparado) / TAXA:.2f} s, {dbfs(aparado):.1f} dBFS) -> {destino.name}")
        gravadas += 1
    return gravadas


def empacotar(pasta: Path) -> Path:
    destino = pasta.with_suffix(".zip")
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as arquivo:
        for wav in sorted(pasta.rglob("*.wav")):
            arquivo.write(wav, wav.relative_to(pasta).as_posix())
    return destino


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--positivas", type=int, default=30, help="quantas vezes dizer 'hey chubasa'")
    parser.add_argument("--negativas", type=int, default=30, help="quantas frases que NÃO são a palavra-chave")
    parser.add_argument("--segundos", type=float, default=2.5, help="duração de cada gravação")
    parser.add_argument("--saida", type=Path, default=PASTA_PADRAO)
    parser.add_argument("--dispositivo", help="microfone (padrão: MIC_DEVICE do .env)")
    args = parser.parse_args(argv)

    if args.positivas < 0 or args.negativas < 0 or args.segundos <= 0:
        parser.error("quantidades não podem ser negativas e a duração tem que ser positiva")

    try:
        mic = args.dispositivo if args.dispositivo is not None else carregar_config().mic_device
        dispositivo = resolver_dispositivo(mic)
    except (ConfiguracaoInvalida, ErroDeAudio) as erro:
        print(erro, file=sys.stderr)
        return 2

    print("Dicas: fale no tom normal, varie um pouco a distância e a velocidade, e grave")
    print("no lugar onde você vai usar o Chubasa. Ctrl+C interrompe (o que já gravou fica salvo).")
    try:
        if args.positivas:
            print(f"\n== Positivas: diga \"hey chubasa\" {args.positivas} vezes ==")
            gravar_serie(args.saida / "positivas", "hey_chubasa", args.positivas, args.segundos,
                         dispositivo, lambda _i: "hey chubasa")
        if args.negativas:
            print(f"\n== Negativas: {args.negativas} frases que NÃO são a palavra-chave ==")
            gravar_serie(args.saida / "negativas", "negativa", args.negativas, args.segundos, dispositivo,
                         lambda i: FRASES_NEGATIVAS[i % len(FRASES_NEGATIVAS)])
    except KeyboardInterrupt:
        print("\nInterrompido.")
    except ErroDeAudio as erro:
        print(f"Erro de áudio: {erro}", file=sys.stderr)
        return 1

    if not args.saida.exists():
        return 0
    pacote = empacotar(args.saida)
    total = len(list(args.saida.rglob("*.wav")))
    print(f"\n{total} amostras em {args.saida}. Pacote para o Colab: {pacote}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
