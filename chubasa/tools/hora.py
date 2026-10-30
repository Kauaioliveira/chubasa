"""Data e hora por extenso, em português (sem depender do locale do sistema)."""

from __future__ import annotations

from datetime import datetime

DIAS = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira", "sábado", "domingo"]
MESES = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro",
         "outubro", "novembro", "dezembro"]


def descrever_agora(agora: datetime | None = None) -> str:
    agora = agora or datetime.now().astimezone()
    minutos = "em ponto" if agora.minute == 0 else f"e {agora.minute} minutos"
    fuso = agora.tzname() or "horário local"
    return (f"Hoje é {DIAS[agora.weekday()]}, {agora.day} de {MESES[agora.month - 1]} de {agora.year}. "
            f"Agora são {agora.hour} horas {minutos} ({agora:%H:%M}, fuso {fuso}).")
