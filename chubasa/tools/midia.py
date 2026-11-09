"""Controle de mídia e volume pelas teclas multimídia do Windows (as mesmas do teclado).

Só estas teclas são enviadas; nada mais de teclado ou mouse. Funciona com qualquer player que
responda às teclas de mídia (Spotify, YouTube no navegador, etc.). Fora do Windows, não suportado.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable

# Códigos virtuais das teclas multimídia (WinUser.h)
VK_VOLUME_MUTE = 0xAD
VK_VOLUME_DOWN = 0xAE
VK_VOLUME_UP = 0xAF
VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1
VK_MEDIA_STOP = 0xB2
VK_MEDIA_PLAY_PAUSE = 0xB3

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002

ACOES_MIDIA = {
    "tocar_pausar": VK_MEDIA_PLAY_PAUSE,
    "proxima": VK_MEDIA_NEXT_TRACK,
    "anterior": VK_MEDIA_PREV_TRACK,
    "parar": VK_MEDIA_STOP,
}
ACOES_VOLUME = {
    "aumentar": VK_VOLUME_UP,
    "diminuir": VK_VOLUME_DOWN,
    "mudo": VK_VOLUME_MUTE,
}
MAXIMO_PASSOS = 25  # cada passo do Windows é 2%: 25 passos = 50%

log = logging.getLogger(__name__)


class ErroMidia(Exception):
    pass


def _keybd_event(vk: int, flags: int) -> None:
    import ctypes

    ctypes.windll.user32.keybd_event(vk, 0, flags, 0)


class TeclasDeMidia:
    def __init__(self, enviar: Callable[[int, int], None] | None = None, plataforma: str = sys.platform,
                 pausa_s: float = 0.03):
        self._enviar = enviar if enviar is not None else _keybd_event
        self.suportado = plataforma == "win32"
        self._pausa_s = pausa_s

    def _apertar(self, vk: int, vezes: int = 1) -> None:
        if not self.suportado:
            raise ErroMidia("controle de mídia e volume só funciona no Windows por enquanto")
        for i in range(vezes):
            self._enviar(vk, KEYEVENTF_EXTENDEDKEY)
            self._enviar(vk, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP)
            if i + 1 < vezes and self._pausa_s:
                time.sleep(self._pausa_s)

    def midia(self, acao: str) -> str:
        vk = ACOES_MIDIA.get((acao or "").strip().lower())
        if vk is None:
            raise ErroMidia(f"ação de mídia desconhecida: {acao!r} (use {', '.join(ACOES_MIDIA)})")
        self._apertar(vk)
        log.info("mídia: %s", acao)
        return acao

    def volume(self, acao: str, passos: int = 5) -> int:
        """Devolve quantos passos foram dados (0 para 'mudo')."""
        acao = (acao or "").strip().lower()
        vk = ACOES_VOLUME.get(acao)
        if vk is None:
            raise ErroMidia(f"ação de volume desconhecida: {acao!r} (use {', '.join(ACOES_VOLUME)})")
        if acao == "mudo":
            self._apertar(vk)
            log.info("volume: mudo (liga/desliga)")
            return 0
        passos = max(1, min(MAXIMO_PASSOS, int(passos)))
        self._apertar(vk, passos)
        log.info("volume: %s %d passo(s)", acao, passos)
        return passos
