"""Abrir sites e pesquisas no navegador padrão. Só http/https; nada de file:, javascript: etc."""

from __future__ import annotations

import logging
import re
import webbrowser
from collections.abc import Callable
from urllib.parse import quote_plus, urlsplit

ESQUEMAS_PERMITIDOS = ("http", "https")
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


class Navegador:
    def __init__(self, abrir: Callable[[str], bool] | None = None):
        self._abrir = abrir if abrir is not None else (lambda url: webbrowser.open(url, new=2))

    def abrir(self, url: str) -> str:
        try:
            ok = self._abrir(url)
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
