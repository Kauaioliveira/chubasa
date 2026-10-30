"""Verificação de voz: confere se quem falou a palavra-chave é a pessoa cadastrada.

Usa o SpeechBrain ECAPA (speechbrain/spkrec-ecapa-voxceleb) em CPU para transformar o áudio numa
"impressão de voz" (embedding) e compara com o perfil cadastrado pela similaridade de cosseno.
O perfil é só um vetor de números: nenhuma gravação é salva.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chubasa.audio import TAXA, aparar_silencio, dbfs, gravar, para_float32

MODELO_ECAPA = "speechbrain/spkrec-ecapa-voxceleb"
MINIMO_FALA_S = 1.0
VOLUME_MINIMO_DBFS = -45.0

FRASES_CADASTRO = [
    "Hey Chubasa, que horas são agora?",
    "O rato roeu a roupa do rei de Roma.",
    "Hey Chubasa, como vai estar o tempo amanhã?",
    "Hoje eu acordei cedo e fui caminhar no parque.",
    "Hey Chubasa, coloca um timer de cinco minutos.",
    "Preciso lembrar de comprar pão e café no mercado.",
    "Hey Chubasa, me conta uma curiosidade.",
    "A reunião de amanhã foi remarcada para as três da tarde.",
    "Hey Chubasa, qual é a capital da Austrália?",
    "Meu jogo favorito tem um mapa enorme para explorar.",
    "Quanto é vinte e sete vezes catorze?",
    "Liga a luz da sala e abaixa o volume da música.",
    "Ontem choveu tanto que a rua ficou alagada.",
    "Eu gosto de programar ouvindo música instrumental.",
    "Hey Chubasa, obrigado, por hoje é só.",
    "Três pratos de trigo para três tigres tristes.",
    "Hey Chubasa, toca alguma coisa animada.",
    "No fim de semana vou visitar a minha família.",
    "O computador ficou lento depois da atualização.",
    "Hey Chubasa, esquece o que eu falei.",
]

Codificador = Callable[[np.ndarray], np.ndarray]
"""Recebe áudio mono 16 kHz (int16 ou float32) e devolve o embedding (vetor 1-D)."""

log = logging.getLogger(__name__)


class ErroVoz(Exception):
    """Modelo de voz indisponível ou perfil de voz inválido."""


@dataclass(frozen=True)
class ResultadoVoz:
    aceito: bool
    score: float | None  # None quando a verificação não rodou
    verificado: bool


# ---------------------------------------------------------------- matemática

def normalizar(vetor: np.ndarray) -> np.ndarray:
    vetor = np.asarray(vetor, dtype=np.float32).ravel()
    norma = float(np.linalg.norm(vetor))
    if norma == 0 or not np.isfinite(norma):
        raise ErroVoz("embedding inválido (vetor nulo ou com NaN)")
    return vetor / norma


def similaridade_cosseno(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(normalizar(a), normalizar(b)))


def criar_perfil(amostras: Sequence[np.ndarray], codificar: Codificador) -> tuple[np.ndarray, list[float]]:
    """Perfil = média dos embeddings normalizados. Devolve também o score de cada amostra contra ele."""
    if not amostras:
        raise ErroVoz("nenhuma amostra para cadastrar")
    embeddings = [normalizar(codificar(amostra)) for amostra in amostras]
    perfil = normalizar(np.mean(embeddings, axis=0))
    return perfil, [float(np.dot(e, perfil)) for e in embeddings]


# ---------------------------------------------------------------- perfil em disco

def salvar_perfil(caminho: Path, perfil: np.ndarray) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    np.save(caminho, normalizar(perfil))


def carregar_perfil(caminho: Path) -> np.ndarray | None:
    if not caminho.is_file():
        return None
    try:
        perfil = np.load(caminho, allow_pickle=False)
        return normalizar(perfil)
    except (OSError, ValueError, ErroVoz) as erro:
        raise ErroVoz(
            f"O perfil de voz {caminho} está corrompido ({erro}). "
            "Apague o arquivo e cadastre de novo: python -m chubasa cadastrar-voz"
        ) from erro


# ---------------------------------------------------------------- modelo ECAPA

def carregar_ecapa(pasta_modelos: Path) -> Codificador:
    try:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy
    except ImportError as erro:
        raise ErroVoz(
            "A verificação de voz precisa das dependências do grupo 'voz': "
            "python -m pip install -e \".[voz]\""
        ) from erro

    # No Windows sem modo desenvolvedor o cache do Hugging Face não cria links; o aviso só polui o log.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    if not log.isEnabledFor(logging.DEBUG):
        logging.getLogger("speechbrain").setLevel(logging.WARNING)
    log.info("Carregando o modelo de voz (%s)...", MODELO_ECAPA)
    try:
        modelo = EncoderClassifier.from_hparams(
            source=MODELO_ECAPA,
            savedir=str(pasta_modelos / "ecapa"),
            run_opts={"device": "cpu"},
            local_strategy=LocalStrategy.COPY,
        )
    except Exception as erro:  # rede, cache corrompido, versão incompatível
        raise ErroVoz(
            f"Não consegui carregar o modelo de voz ({erro}). Na primeira vez ele é baixado da "
            "internet (~80 MB); confira a conexão."
        ) from erro

    def codificar(audio: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            entrada = torch.from_numpy(para_float32(audio)).unsqueeze(0)
            return modelo.encode_batch(entrada).squeeze().cpu().numpy().astype(np.float32)

    return codificar


# ---------------------------------------------------------------- verificação

class VerificadorVoz:
    """Compara o áudio da palavra-chave + comando com o perfil cadastrado.

    Sem perfil cadastrado (ou com SPEAKER_VERIFY=false) aceita qualquer voz e avisa no log,
    para o assistente funcionar antes do cadastro.
    """

    def __init__(self, perfil: np.ndarray | None, limiar: float,
                 carregar_codificador: Callable[[], Codificador], ligado: bool = True):
        self.perfil = perfil
        self.limiar = limiar
        self.ligado = ligado
        self._carregar_codificador = carregar_codificador
        self._codificar: Codificador | None = None
        if not ligado:
            log.warning("Verificação de voz desligada (SPEAKER_VERIFY=false): qualquer voz ativa o Chubasa.")
        elif perfil is None:
            log.warning("Nenhuma voz cadastrada: qualquer voz ativa o Chubasa. "
                        "Cadastre a sua com: python -m chubasa cadastrar-voz")

    @property
    def ativo(self) -> bool:
        return self.ligado and self.perfil is not None

    def preparar(self) -> None:
        """Carrega o modelo agora (leva alguns segundos), em vez de na primeira ativação."""
        if self.ativo and self._codificar is None:
            self._codificar = self._carregar_codificador()

    def verificar(self, audio_palavra: np.ndarray, audio_comando: np.ndarray | None = None) -> ResultadoVoz:
        if not self.ativo:
            return ResultadoVoz(aceito=True, score=None, verificado=False)
        partes = [audio_palavra] if audio_comando is None else [audio_palavra, audio_comando]
        trecho = np.concatenate([para_float32(p) for p in partes])
        if trecho.size == 0:
            log.info("voz: sem áudio para verificar -> rejeitada")
            return ResultadoVoz(aceito=False, score=None, verificado=True)

        self.preparar()
        score = similaridade_cosseno(self._codificar(trecho), self.perfil)
        aceito = score >= self.limiar
        log.info("voz: score=%.3f limiar=%.2f -> %s", score, self.limiar, "aceita" if aceito else "rejeitada")
        return ResultadoVoz(aceito=aceito, score=score, verificado=True)


def criar_verificador(config, carregar: Callable[[Path], Codificador] = carregar_ecapa) -> VerificadorVoz:
    perfil = carregar_perfil(config.speaker_profile_path) if config.speaker_verify else None
    return VerificadorVoz(perfil, config.speaker_threshold, lambda: carregar(config.models_dir),
                          ligado=config.speaker_verify)


# ---------------------------------------------------------------- cadastro

def problema_na_frase(aparado: np.ndarray) -> str | None:
    if aparado.size < MINIMO_FALA_S * TAXA:
        return "ficou curto demais ou não ouvi fala"
    if dbfs(aparado) < VOLUME_MINIMO_DBFS:
        return "ficou baixo demais, chegue mais perto do microfone"
    return None


def comando_cadastrar_voz(config, dispositivo: int | None, n_frases: int = 15, segundos: float = 5.0,
                          carregar: Callable[[Path], Codificador] = carregar_ecapa,
                          gravar_audio: Callable[[float, int | None], np.ndarray] = gravar,
                          perguntar: Callable[[str], str] = input) -> int:
    if not 10 <= n_frases <= len(FRASES_CADASTRO):
        print(f"Use entre 10 e {len(FRASES_CADASTRO)} frases.", file=sys.stderr)
        return 2

    codificar = carregar(config.models_dir)  # falha cedo, antes de pedir para falar
    print(f"Cadastro de voz: leia {n_frases} frases em voz alta, no tom normal, no lugar onde vai usar o")
    print("Chubasa. Cada gravação dura até", f"{segundos:.0f} s. Nada é salvo além da impressão de voz.")

    amostras: list[np.ndarray] = []
    while len(amostras) < n_frases:
        frase = FRASES_CADASTRO[len(amostras)]
        perguntar(f"\n[{len(amostras) + 1}/{n_frases}] Enter e leia: \"{frase}\"")
        print("  gravando...")
        aparado = aparar_silencio(gravar_audio(segundos, dispositivo))
        problema = problema_na_frase(aparado)
        if problema:
            print(f"  {problema}. Vamos de novo.")
            continue
        amostras.append(aparado)
        print(f"  ok ({len(aparado) / TAXA:.1f} s)")

    perfil, scores = criar_perfil(amostras, codificar)
    salvar_perfil(config.speaker_profile_path, perfil)
    print(f"\nVoz cadastrada em {config.speaker_profile_path}")
    print(f"Coerência das frases com o perfil: mín {min(scores):.2f}, média {np.mean(scores):.2f} "
          f"(limiar atual SPEAKER_THRESHOLD={config.speaker_threshold:.2f})")
    destoantes = [i + 1 for i, s in enumerate(scores) if s < np.mean(scores) - 0.15]
    if destoantes:
        print(f"As frases {destoantes} destoaram das outras (ruído? outra pessoa?). "
              "Se o reconhecimento falhar, cadastre de novo num lugar mais silencioso.")
    return 0
