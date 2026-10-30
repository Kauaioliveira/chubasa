"""Sinais de áudio sintéticos para os testes (nada de microfone nem gravação real)."""

from __future__ import annotations

import numpy as np

from chubasa import audio


def seno(segundos: float, freq: float = 440.0, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(segundos * audio.TAXA)) / audio.TAXA
    return audio.para_int16(amplitude * np.sin(2 * np.pi * freq * t))


def fala_sintetica(antes_s=0.8, fala_s=0.6, depois_s=0.9, amplitude=0.3, semente=1) -> np.ndarray:
    """Silêncio com um chiado bem baixo, um tom no meio (a 'fala') e silêncio de novo."""
    gerador = np.random.default_rng(semente)
    t = np.arange(int(fala_s * audio.TAXA)) / audio.TAXA
    partes = [
        gerador.normal(0, 0.0005, int(antes_s * audio.TAXA)),
        amplitude * np.sin(2 * np.pi * 220 * t),
        gerador.normal(0, 0.0005, int(depois_s * audio.TAXA)),
    ]
    return audio.para_int16(np.concatenate(partes))
