"""Fala para texto com faster-whisper, em CPU (int8) por padrão, idioma fixo."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from chubasa.audio import TAXA, para_float32

# Segmento com chance alta de "não é fala" costuma ser alucinação do Whisper ("Legendas por...").
LIMITE_SEM_FALA = 0.6

log = logging.getLogger(__name__)


class ErroSTT(Exception):
    """Modelo de transcrição indisponível."""


@dataclass(frozen=True)
class Transcricao:
    texto: str
    segundos_audio: float
    segundos_processamento: float


def carregar_whisper(config) -> Any:
    from faster_whisper import WhisperModel

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    log.info("Carregando o Whisper '%s' (%s, %s); na primeira vez ele é baixado...",
             config.stt_model, config.stt_device, config.stt_compute_type)
    inicio = time.perf_counter()
    try:
        modelo = WhisperModel(
            config.stt_model,
            device=config.stt_device,
            compute_type=config.stt_compute_type,
            cpu_threads=config.stt_cpu_threads,
            download_root=str(config.models_dir / "whisper"),
        )
    except RuntimeError as erro:
        if config.stt_device != "cpu":
            raise ErroSTT(f"Não consegui usar a GPU ({erro}). Use STT_DEVICE=cpu no .env.") from erro
        raise ErroSTT(f"Não consegui carregar o Whisper ({erro}).") from erro
    except Exception as erro:  # rede/Hugging Face/disco
        raise ErroSTT(
            f"Não consegui baixar/carregar o Whisper '{config.stt_model}' ({erro}). "
            "Confira a internet (só na primeira vez) e o nome em STT_MODEL."
        ) from erro
    log.info("Whisper pronto em %.1f s", time.perf_counter() - inicio)
    return modelo


class Transcritor:
    """Carrega o modelo uma vez e transcreve cada comando."""

    def __init__(self, modelo: Any, idioma: str, beam_size: int = 1):
        self._modelo = modelo
        self.idioma = idioma
        self.beam_size = beam_size

    def transcrever(self, audio: np.ndarray) -> Transcricao:
        amostras = para_float32(audio)
        duracao = len(amostras) / TAXA
        inicio = time.perf_counter()
        segmentos, _info = self._modelo.transcribe(
            amostras,
            language=self.idioma,
            beam_size=self.beam_size,
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=False,  # o VAD do Bloco 3 já cortou o comando
        )
        # `segmentos` é um gerador: a transcrição de fato acontece aqui.
        partes = [s.text.strip() for s in segmentos if s.no_speech_prob < LIMITE_SEM_FALA]
        texto = " ".join(p for p in partes if p)
        gasto = time.perf_counter() - inicio
        log.info("transcrição: %.2f s de áudio em %.2f s -> %r", duracao, gasto, texto)
        return Transcricao(texto, duracao, gasto)


def criar_transcritor(config, carregar: Callable[[Any], Any] = carregar_whisper) -> Transcritor:
    return Transcritor(carregar(config), config.language, config.stt_beam_size)


def comando_transcrever(config, dispositivo: int | None, arquivo=None,
                        criar: Callable[[Any], Transcritor] = criar_transcritor) -> int:
    """Transcreve um WAV de 16 kHz ou, sem arquivo, um comando falado depois do bip."""
    from chubasa.audio import Microfone, ler_wav

    if arquivo is not None:
        audio, taxa = ler_wav(arquivo)
        if taxa != TAXA:
            print(f"{arquivo}: o WAV precisa ser de {TAXA} Hz (este é de {taxa} Hz).")
            return 2
        transcritor = criar(config)
    else:
        from chubasa.vad import criar_gravador
        from chubasa.wakeword import tocar_bip

        transcritor = criar(config)  # carrega antes de pedir para falar
        gravador = criar_gravador(config)
        print("Fale depois do bip.", flush=True)
        tocar_bip()
        with Microfone(dispositivo) as mic:
            resultado = gravador.gravar(mic.frames(), np.zeros(0, dtype=np.int16))
        if not resultado.teve_fala:
            print("Não ouvi nada.")
            return 1
        audio = resultado.audio_comando

    transcricao = transcritor.transcrever(audio)
    print(transcricao.texto or "(nada reconhecido)")
    print(f"{transcricao.segundos_audio:.1f} s de áudio transcritos em {transcricao.segundos_processamento:.2f} s")
    return 0
