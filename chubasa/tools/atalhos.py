"""Atalhos do Menu Iniciar do Windows: abrir programas instalados pelo nome.

Só abre atalhos (.lnk) encontrados nas pastas do Menu Iniciar; o caminho nunca vem do modelo.
Desinstaladores ficam de fora. Se o nome bater com mais de um programa, devolve as opções.
"""

from __future__ import annotations

import difflib
import logging
import os
import re
import time
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

# Desinstaladores e ferramentas de sistema que apagam ou alteram coisas: nunca abrir por voz
_PROIBIDOS = re.compile(
    r"uninstall|desinstal|remov|unins\d*\b|cleanup|limpeza de disco|dfrgui|desfragment|defrag"
    r"|registry editor|editor do registro|regedit|recovery|recupera|system restore|restaura"
    r"|system configuration|configura..o do sistema|msconfig|reset",
    re.IGNORECASE,
)
CACHE_S = 300
MAXIMO_OPCOES = 5

log = logging.getLogger(__name__)


def chave(nome: str) -> str:
    """'Microsoft Word (x64)' -> 'microsoft word x64'; sem acento, sem símbolos, minúsculas."""
    sem_acento = unicodedata.normalize("NFKD", nome).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", sem_acento.lower()).split())


@dataclass(frozen=True)
class Atalho:
    nome: str  # como aparece no Menu Iniciar
    caminho: Path


class ProgramaAmbiguo(Exception):
    def __init__(self, pedido: str, opcoes: list[str]):
        super().__init__(f"'{pedido}' combina com: {', '.join(opcoes)}")
        self.pedido = pedido
        self.opcoes = opcoes


def pastas_do_menu_iniciar(ambiente: dict[str, str] | None = None) -> list[Path]:
    ambiente = os.environ if ambiente is None else ambiente
    todos = (ambiente.get("ProgramData") or ambiente.get("PROGRAMDATA") or ambiente.get("ALLUSERSPROFILE")
             or str(Path(ambiente.get("SystemDrive", "C:") + "\\") / "ProgramData"))
    pastas = [Path(todos) / "Microsoft" / "Windows" / "Start Menu" / "Programs"]
    usuario = ambiente.get("APPDATA")
    if usuario:
        pastas.append(Path(usuario) / "Microsoft" / "Windows" / "Start Menu" / "Programs")
    return pastas


def varrer(pastas: Iterable[Path]) -> list[Atalho]:
    atalhos: dict[str, Atalho] = {}
    for pasta in pastas:
        if not pasta.is_dir():
            continue
        for arquivo in pasta.rglob("*.lnk"):
            nome = arquivo.stem.strip()
            if not nome or _PROIBIDOS.search(nome):
                continue
            atalhos.setdefault(chave(nome), Atalho(nome, arquivo))  # mesmo nome nas duas pastas: o 1º
    return sorted(atalhos.values(), key=lambda a: chave(a.nome))


class AtalhosDoMenuIniciar:
    def __init__(self, listar: Callable[[], list[Atalho]] | None = None, relogio: Callable[[], float] = time.monotonic):
        self._listar = listar if listar is not None else (lambda: varrer(pastas_do_menu_iniciar()))
        self._relogio = relogio
        self._cache: list[Atalho] = []
        self._quando: float | None = None

    def todos(self) -> list[Atalho]:
        agora = self._relogio()
        if self._quando is None or agora - self._quando > CACHE_S:  # programa instalado agora aparece
            self._cache = self._listar()
            self._quando = agora
        return self._cache

    def procurar(self, pedido: str) -> Atalho | None:
        """O atalho que corresponde ao pedido; None se não achar; ProgramaAmbiguo se houver vários."""
        alvo = chave(pedido)
        if not alvo:
            return None
        atalhos = self.todos()
        por_chave = {chave(a.nome): a for a in atalhos}

        if alvo in por_chave:
            return por_chave[alvo]

        palavras = alvo.split()
        contem = [a for k, a in por_chave.items() if all(p in k.split() for p in palavras)]
        if not contem:  # prefixo de palavra: "photo" -> "Photoshop"
            contem = [a for k, a in por_chave.items()
                      if all(any(w.startswith(p) for w in k.split()) for p in palavras)]
        if len(contem) == 1:
            return contem[0]
        if len(contem) > 1:
            # nome mais curto primeiro: "Word" antes de "Word Mobile Viewer"
            contem.sort(key=lambda a: len(chave(a.nome)))
            raise ProgramaAmbiguo(pedido, [a.nome for a in contem[:MAXIMO_OPCOES]])

        parecidos = difflib.get_close_matches(alvo, list(por_chave), n=MAXIMO_OPCOES, cutoff=0.75)
        if len(parecidos) == 1:
            return por_chave[parecidos[0]]
        if parecidos:
            melhor, segundo = (difflib.SequenceMatcher(None, alvo, k).ratio() for k in parecidos[:2])
            if melhor - segundo >= 0.1:
                return por_chave[parecidos[0]]
            raise ProgramaAmbiguo(pedido, [por_chave[k].nome for k in parecidos])
        return None
