"""Gravação do comando depois da palavra-chave, com detecção de fala Silero VAD (ONNX, sem torch)."""

from __future__ import annotations

import hashlib
import logging
import math
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from chubasa.audio import DURACAO_FRAME, TAXA

# Versão fixada do modelo, conferida pelo SHA-256 depois do download
URL_SILERO = "https://github.com/snakers4/silero-vad/raw/v6.2.3/src/silero_vad/data/silero_vad.onnx"
SHA256_SILERO = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
ARQUIVO_SILERO = "silero_vad.onnx"

AMOSTRAS_POR_JANELA = 512  # o Silero a 16 kHz trabalha em janelas de 32 ms
AMOSTRAS_DE_CONTEXTO = 64  # e precisa do fim da janela anterior como contexto

log = logging.getLogger(__name__)


class ErroVAD(Exception):
    """Modelo do VAD indisponível."""


class DetectorDeFala(Protocol):
    def probabilidade(self, frame: np.ndarray) -> float: ...
    def reset(self) -> None: ...


# ---------------------------------------------------------------- modelo

def baixar_silero(pasta: Path) -> Path:
    """Baixa o modelo Silero VAD (~2 MB) uma vez e confere a integridade."""
    destino = pasta / ARQUIVO_SILERO
    if destino.is_file():
        return destino
    import requests

    log.info("Baixando o Silero VAD para %s (só na primeira vez)", destino)
    pasta.mkdir(parents=True, exist_ok=True)
    try:
        resposta = requests.get(URL_SILERO, timeout=60)
        resposta.raise_for_status()
    except requests.RequestException as erro:
        raise ErroVAD(f"Não consegui baixar o Silero VAD ({erro}). Confira a conexão com a internet.") from erro
    if hashlib.sha256(resposta.content).hexdigest() != SHA256_SILERO:
        raise ErroVAD("O arquivo do Silero VAD baixado não confere com o esperado (SHA-256). Tente de novo.")
    temporario = destino.with_suffix(".parcial")
    temporario.write_bytes(resposta.content)
    temporario.replace(destino)
    return destino


class SileroVAD:
    """Silero VAD v6 rodando direto no onnxruntime. Aceita frames de qualquer tamanho (int16 ou float32)."""

    def __init__(self, caminho: Path):
        import onnxruntime as ort

        opcoes = ort.SessionOptions()
        opcoes.inter_op_num_threads = 1
        opcoes.intra_op_num_threads = 1
        try:
            self._sessao = ort.InferenceSession(str(caminho), sess_options=opcoes,
                                                providers=["CPUExecutionProvider"])
        except Exception as erro:  # onnxruntime levanta tipos variados para arquivo inválido
            raise ErroVAD(f"Não consegui carregar o Silero VAD em {caminho}: {erro}. Apague o arquivo e rode de novo.") \
                from erro
        self._taxa = np.array(TAXA, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._estado = np.zeros((2, 1, 128), dtype=np.float32)
        self._contexto = np.zeros(AMOSTRAS_DE_CONTEXTO, dtype=np.float32)
        self._pendente = np.zeros(0, dtype=np.float32)
        self._ultima = 0.0

    def probabilidade(self, frame: np.ndarray) -> float:
        """Maior probabilidade de fala entre as janelas completadas por este frame."""
        amostras = frame.astype(np.float32) / 32768.0 if frame.dtype == np.int16 else frame.astype(np.float32)
        self._pendente = np.concatenate([self._pendente, amostras])
        probabilidades = []
        while len(self._pendente) >= AMOSTRAS_POR_JANELA:
            janela = self._pendente[:AMOSTRAS_POR_JANELA]
            self._pendente = self._pendente[AMOSTRAS_POR_JANELA:]
            entrada = np.concatenate([self._contexto, janela])[np.newaxis, :]
            saida, self._estado = self._sessao.run(None, {"input": entrada, "state": self._estado, "sr": self._taxa})
            self._contexto = janela[-AMOSTRAS_DE_CONTEXTO:]
            probabilidades.append(float(saida[0, 0]))
        if probabilidades:
            self._ultima = max(probabilidades)
        return self._ultima


def criar_vad(config) -> SileroVAD:
    return SileroVAD(baixar_silero(config.models_dir / "silero"))


# ---------------------------------------------------------------- gravação do comando

@dataclass(frozen=True)
class ResultadoGravacao:
    motivo: str  # "fim_de_fala", "sem_fala" ou "limite"
    audio_palavra: np.ndarray  # trecho da palavra-chave (para a verificação de voz)
    audio_comando: np.ndarray  # pré-roll + fala, sem o silêncio final longo

    @property
    def teve_fala(self) -> bool:
        return self.motivo != "sem_fala"

    @property
    def duracao_comando(self) -> float:
        return len(self.audio_comando) / TAXA


class GravadorComando:
    """Grava o comando depois do bip: espera a fala começar, para após um silêncio.

    - Pré-roll: os últimos `preroll_s` antes da fala entram na gravação (não corta a 1ª sílaba).
    - Para depois de `silencio_fim_s` sem fala, mantendo só `cauda_s` desse silêncio.
    - Desiste se ninguém falar em `timeout_sem_fala_s`; corta em `maximo_s`.
    - Histerese: a fala começa com prob >= limiar e só conta como silêncio abaixo de limiar - 0,15.
    """

    def __init__(self, vad: DetectorDeFala, limiar: float = 0.5, silencio_fim_s: float = 1.0,
                 timeout_sem_fala_s: float = 5.0, maximo_s: float = 30.0, preroll_s: float = 0.3,
                 cauda_s: float = 0.25, inicio_minimo_s: float = 0.16):
        self._vad = vad
        self.limiar = limiar
        self.limiar_silencio = max(0.0, limiar - 0.15)
        self._frames_silencio_fim = max(1, math.ceil(silencio_fim_s / DURACAO_FRAME))
        self._frames_timeout = max(1, math.ceil(timeout_sem_fala_s / DURACAO_FRAME))
        self._frames_maximo = max(1, math.ceil(maximo_s / DURACAO_FRAME))
        self._frames_preroll = max(0, math.ceil(preroll_s / DURACAO_FRAME))
        self._frames_cauda = max(0, math.ceil(cauda_s / DURACAO_FRAME))
        self._frames_inicio = max(1, math.ceil(inicio_minimo_s / DURACAO_FRAME))

    def gravar(self, frames: Iterator[np.ndarray], audio_palavra: np.ndarray,
               timeout_sem_fala_s: float | None = None) -> ResultadoGravacao:
        """`timeout_sem_fala_s` substitui o padrão nesta gravação (ex.: o modo conversa)."""
        self._vad.reset()
        frames_timeout = (self._frames_timeout if timeout_sem_fala_s is None
                          else max(1, math.ceil(timeout_sem_fala_s / DURACAO_FRAME)))
        espera: deque[np.ndarray] = deque(maxlen=self._frames_preroll + self._frames_inicio)
        seguidos = 0
        comando: list[np.ndarray] | None = None

        for indice, frame in enumerate(frames):
            prob = self._vad.probabilidade(frame)
            if comando is None:
                espera.append(frame)
                seguidos = seguidos + 1 if prob >= self.limiar else 0
                if seguidos >= self._frames_inicio:
                    comando = list(espera)  # pré-roll + os frames que confirmaram a fala
                    silencio = 0
                    log.debug("fala começou após %.2f s", (indice + 1) * DURACAO_FRAME)
                elif indice + 1 >= frames_timeout:
                    log.info("ninguém falou em %.1f s", frames_timeout * DURACAO_FRAME)
                    return ResultadoGravacao("sem_fala", audio_palavra, np.zeros(0, dtype=np.int16))
                continue

            comando.append(frame)
            silencio = silencio + 1 if prob < self.limiar_silencio else 0
            if silencio >= self._frames_silencio_fim:
                corte = len(comando) - silencio + self._frames_cauda
                return self._resultado("fim_de_fala", audio_palavra, comando[:corte])
            if len(comando) >= self._frames_maximo:
                log.info("comando chegou ao limite de %.0f s", self._frames_maximo * DURACAO_FRAME)
                return self._resultado("limite", audio_palavra, comando)

        # Fonte de áudio acabou (microfone fechado/arquivo): devolve o que tiver.
        if comando is None:
            return ResultadoGravacao("sem_fala", audio_palavra, np.zeros(0, dtype=np.int16))
        return self._resultado("fim_de_fala", audio_palavra, comando)

    @staticmethod
    def _resultado(motivo: str, audio_palavra: np.ndarray, frames: list[np.ndarray]) -> ResultadoGravacao:
        audio = np.concatenate(frames).astype(np.int16)
        log.info("comando gravado: %.2f s (%s)", len(audio) / TAXA, motivo)
        return ResultadoGravacao(motivo, audio_palavra, audio)


def criar_gravador(config, vad: DetectorDeFala | None = None) -> GravadorComando:
    return GravadorComando(
        vad if vad is not None else criar_vad(config),
        limiar=config.vad_threshold,
        silencio_fim_s=config.vad_silence_end_s,
        timeout_sem_fala_s=config.vad_no_speech_timeout_s,
        maximo_s=config.vad_max_recording_s,
        preroll_s=config.vad_preroll_s,
    )
