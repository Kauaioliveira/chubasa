"""Timers e lembretes que avisam por voz, em threads próprias (não travam o loop)."""

from __future__ import annotations

import itertools
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

MAXIMO_S = 24 * 3600
MAXIMO_ATIVOS = 10

log = logging.getLogger(__name__)


class ErroTimer(Exception):
    pass


def falar_duracao(segundos: float) -> str:
    """'5 minutos', '1 hora e 30 minutos', '45 segundos' (arredondado como se fala)."""
    total = max(0, int(round(segundos)))
    horas, resto = divmod(total, 3600)
    minutos, segs = divmod(resto, 60)
    partes = []
    if horas:
        partes.append(f"{horas} hora" + ("s" if horas > 1 else ""))
    if minutos:
        partes.append(f"{minutos} minuto" + ("s" if minutos > 1 else ""))
    if segs and not horas:
        partes.append(f"{segs} segundo" + ("s" if segs > 1 else ""))
    if not partes:
        return "0 segundos"
    return " e ".join(partes)


@dataclass
class Timer:
    id: int
    duracao_s: int
    lembrete: str
    fim: float
    _thread: threading.Timer | None = field(default=None, repr=False)

    def restante_s(self) -> float:
        return max(0.0, self.fim - time.monotonic())


class Timers:
    def __init__(self, avisar: Callable[[str], None]):
        self._avisar = avisar
        self._ativos: dict[int, Timer] = {}
        self._ids = itertools.count(1)
        self._trava = threading.Lock()

    def criar(self, duracao_s: int, lembrete: str = "") -> Timer:
        if duracao_s < 1:
            raise ErroTimer("a duração precisa ser de pelo menos 1 segundo")
        if duracao_s > MAXIMO_S:
            raise ErroTimer("o máximo é 24 horas")
        with self._trava:
            if len(self._ativos) >= MAXIMO_ATIVOS:
                raise ErroTimer(f"já existem {MAXIMO_ATIVOS} timers ativos")
            timer = Timer(next(self._ids), duracao_s, lembrete.strip(), time.monotonic() + duracao_s)
            thread = threading.Timer(duracao_s, self._disparar, args=(timer.id,))
            thread.daemon = True
            thread.name = f"timer-{timer.id}"
            timer._thread = thread
            self._ativos[timer.id] = timer
        thread.start()
        log.info("timer %d criado: %s%s", timer.id, falar_duracao(duracao_s),
                 f" ({timer.lembrete})" if timer.lembrete else "")
        return timer

    def ativos(self) -> list[Timer]:
        with self._trava:
            return sorted(self._ativos.values(), key=lambda t: t.fim)

    def cancelar(self, id_timer: int) -> bool:
        with self._trava:
            timer = self._ativos.pop(id_timer, None)
        if timer is None:
            return False
        if timer._thread is not None:
            timer._thread.cancel()
        log.info("timer %d cancelado", id_timer)
        return True

    def cancelar_todos(self) -> None:
        for timer in self.ativos():
            self.cancelar(timer.id)

    def _disparar(self, id_timer: int) -> None:
        with self._trava:
            timer = self._ativos.pop(id_timer, None)
        if timer is None:
            return  # cancelado em cima da hora
        if timer.lembrete:
            texto = f"Lembrete: {timer.lembrete}."
        else:
            texto = f"O timer de {falar_duracao(timer.duracao_s)} acabou."
        log.info("timer %d disparou", id_timer)
        try:
            self._avisar(texto)
        except Exception as erro:  # aviso que falha não pode matar outros timers
            log.error("não consegui avisar o timer %d: %s", id_timer, erro)
