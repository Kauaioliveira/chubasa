"""Abrir sites e pesquisas no navegador padrão. Só http/https; nada de file:, javascript: etc."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urlsplit

ESQUEMAS_PERMITIDOS = ("http", "https")
# TOOLS_BROWSER: "padrao" segue o navegador padrão do Windows; os outros abrem sempre aquele.
NAVEGADORES = {"edge": "msedge.exe", "chrome": "chrome.exe", "firefox": "firefox.exe"}
OPCOES_NAVEGADOR = ("padrao", *NAVEGADORES)
# "algo:" no começo que não é host:porta (ex.: javascript:, file:, mailto:, data:)
_ESQUEMA = re.compile(r"^[a-z][a-z0-9+.\-]*:", re.IGNORECASE)
_HOST_E_PORTA = re.compile(r"^[\w.\-]+:\d+(/|$)")

log = logging.getLogger(__name__)


class ErroNavegador(Exception):
    pass


def normalizar_url(endereco: str) -> str:
    """'wikipedia.org' -> 'https://wikipedia.org'; recusa esquemas que não sejam http/https."""
    texto = (endereco or "").strip()
    if not texto:
        raise ErroNavegador("o endereço veio vazio")
    if any(c.isspace() for c in texto):
        raise ErroNavegador(f"'{texto}' não é um endereço de site (tem espaços); para buscar, use a pesquisa")
    if "://" not in texto:
        if _ESQUEMA.match(texto) and not _HOST_E_PORTA.match(texto):
            raise ErroNavegador(f"só abro sites http ou https, não '{texto.split(':', 1)[0]}:'")
        texto = "https://" + texto.lstrip("/")
    partes = urlsplit(texto)
    if partes.scheme.lower() not in ESQUEMAS_PERMITIDOS:
        raise ErroNavegador(f"só abro sites http ou https, não '{partes.scheme}:'")
    host = partes.hostname or ""
    if not host or ("." not in host and host != "localhost"):
        raise ErroNavegador(f"'{endereco}' não parece um endereço de site")
    return texto


def url_google(termo: str) -> str:
    return "https://www.google.com/search?q=" + quote_plus(_termo(termo))


def url_youtube(termo: str) -> str:
    return "https://www.youtube.com/results?search_query=" + quote_plus(_termo(termo))


def _termo(termo: str) -> str:
    termo = (termo or "").strip()
    if not termo:
        raise ErroNavegador("o termo da pesquisa veio vazio")
    return termo


def _caminho_app_paths(exe: str) -> str | None:
    """Caminho registrado em App Paths (onde o Windows acha o msedge.exe, chrome.exe...)."""
    import winreg

    chave = rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{exe}"
    for raiz in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            with winreg.OpenKey(raiz, chave) as k:
                caminho = winreg.QueryValue(k, None)
        except OSError:
            continue
        if caminho and Path(caminho.strip('"')).exists():
            return caminho.strip('"')
    return None


def localizar_navegador(nome: str, app_paths: Callable[[str], str | None] | None = None) -> str | None:
    exe = NAVEGADORES[nome]
    if app_paths is None:
        app_paths = _caminho_app_paths if sys.platform == "win32" else (lambda _exe: None)
    return app_paths(exe) or shutil.which(exe) or shutil.which(Path(exe).stem)


def abridor(nome: str = "padrao", localizar: Callable[[str], str | None] = localizar_navegador,
            iniciar: Callable[[list[str]], Any] | None = None) -> Callable[[str], bool]:
    """Função que abre uma URL no navegador escolhido em TOOLS_BROWSER."""
    nome = (nome or "padrao").strip().lower()
    if nome not in OPCOES_NAVEGADOR:
        raise ValueError(f"TOOLS_BROWSER inválido: {nome!r} (use {', '.join(OPCOES_NAVEGADOR)})")
    if nome == "padrao":
        if sys.platform == "win32":
            # O webbrowser do Python prefere o Chrome quando acha um instalado; o shell segue o padrão do Windows.
            return lambda url: os.startfile(url) or True  # noqa: S606 - url já validada (http/https)
        return lambda url: webbrowser.open(url, new=2)
    iniciar = iniciar if iniciar is not None else (lambda args: subprocess.Popen(args))  # noqa: S603

    def abrir(url: str) -> bool:
        exe = localizar(nome)
        if exe is None:
            raise ErroNavegador(f"não achei o {nome} instalado")
        iniciar([exe, url])
        return True

    return abrir


class Navegador:
    def __init__(self, abrir: Callable[[str], bool] | None = None):
        self._abrir = abrir if abrir is not None else abridor("padrao")

    def abrir(self, url: str) -> str:
        try:
            ok = self._abrir(url)
        except ErroNavegador:
            raise
        except Exception as erro:  # webbrowser/os.startfile podem levantar OSError e afins
            raise ErroNavegador(f"não consegui abrir o navegador ({erro})") from erro
        if ok is False:
            raise ErroNavegador("não achei um navegador para abrir")
        log.info("navegador: %s", url)
        return url

    def abrir_site(self, endereco: str) -> str:
        return self.abrir(normalizar_url(endereco))

    def pesquisar_no_google(self, termo: str) -> str:
        return self.abrir(url_google(termo))

    def pesquisar_no_youtube(self, termo: str) -> str:
        return self.abrir(url_youtube(termo))
