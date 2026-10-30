"""Abrir programas: só os da lista do `.env` (TOOLS_PROGRAMS). O modelo escolhe um nome, nunca um caminho."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import unicodedata
from collections.abc import Callable

log = logging.getLogger(__name__)


class ErroPrograma(Exception):
    pass


def _chave(nome: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", nome).encode("ascii", "ignore").decode()
    return " ".join(sem_acento.lower().split())


def _iniciar(comando: str) -> None:
    if sys.platform == "win32":
        # ShellExecute acha programas registrados (chrome.exe, notepad.exe) sem passar por um shell.
        os.startfile(comando)  # noqa: S606 - comando vem só da lista do .env
    else:
        subprocess.Popen([comando], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,  # noqa: S603
                         start_new_session=True)


class ProgramasPermitidos:
    def __init__(self, programas: dict[str, str], iniciar: Callable[[str], None] = _iniciar):
        self._programas = {_chave(nome): (nome.strip(), comando.strip()) for nome, comando in programas.items()}
        self._iniciar = iniciar

    @classmethod
    def do_texto(cls, texto: str, iniciar: Callable[[str], None] = _iniciar) -> ProgramasPermitidos:
        """Formato do .env: 'nome:comando;nome:comando' (ex.: 'calculadora:calc.exe;bloco de notas:notepad.exe')."""
        programas = {}
        for item in texto.split(";"):
            if not item.strip():
                continue
            nome, separador, comando = item.partition(":")
            if not separador or not nome.strip() or not comando.strip():
                raise ValueError(f"item inválido em TOOLS_PROGRAMS: {item!r} (use nome:comando)")
            programas[nome] = comando
        return cls(programas, iniciar)

    def nomes(self) -> list[str]:
        return [nome for nome, _ in self._programas.values()]

    def abrir(self, nome: str) -> str:
        item = self._programas.get(_chave(nome))
        if item is None:
            raise ErroPrograma(f"'{nome}' não está na lista de programas permitidos ({', '.join(self.nomes())}).")
        nome_exibido, comando = item
        try:
            self._iniciar(comando)
        except OSError as erro:
            raise ErroPrograma(f"Não consegui abrir {nome_exibido}: {erro}") from erro
        log.info("programa aberto: %s (%s)", nome_exibido, comando)
        return nome_exibido
