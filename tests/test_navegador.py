from __future__ import annotations

import pytest

from chubasa.tools.navegador import ErroNavegador, Navegador, normalizar_url, url_google, url_youtube


@pytest.mark.parametrize("entrada, esperado", [
    ("wikipedia.org", "https://wikipedia.org"),
    ("pt.wikipedia.org/wiki/Brasil", "https://pt.wikipedia.org/wiki/Brasil"),
    ("  https://www.google.com  ", "https://www.google.com"),
    ("http://exemplo.com.br/a?b=1", "http://exemplo.com.br/a?b=1"),
    ("HTTPS://Exemplo.com", "HTTPS://Exemplo.com"),
    ("localhost:8000", "https://localhost:8000"),
    ("192.168.0.1", "https://192.168.0.1"),
])
def test_normaliza_endereco(entrada, esperado):
    assert normalizar_url(entrada) == esperado


@pytest.mark.parametrize("entrada", [
    "file:///C:/Windows/win.ini", "javascript:alert(1)", "JavaScript:alert(1)", "data:text/html,oi",
    "mailto:alguem@exemplo.com", "ftp://exemplo.com", "ms-settings:privacy", "vbscript:msgbox",
    r"C:\Windows\System32\cmd.exe", "", "   ", "tempo hoje", "naoeumsite",
])
def test_recusa_o_que_nao_e_site_http(entrada):
    with pytest.raises(ErroNavegador):
        normalizar_url(entrada)


def test_urls_de_pesquisa_codificadas():
    assert url_google("preço do dólar hoje") == "https://www.google.com/search?q=pre%C3%A7o+do+d%C3%B3lar+hoje"
    assert url_youtube("a&b=c") == "https://www.youtube.com/results?search_query=a%26b%3Dc"
    with pytest.raises(ErroNavegador):
        url_google("  ")


def test_abre_no_navegador_padrao():
    abertos = []
    navegador = Navegador(abertos.append)
    navegador.abrir_site("wikipedia.org")
    navegador.pesquisar_no_youtube("chuva")
    assert abertos == ["https://wikipedia.org", "https://www.youtube.com/results?search_query=chuva"]


def test_sem_navegador_da_erro_claro():
    with pytest.raises(ErroNavegador, match="navegador"):
        Navegador(lambda _url: False).abrir_site("wikipedia.org")

    def quebra(_url):
        raise OSError("sem associação")

    with pytest.raises(ErroNavegador, match="sem associação"):
        Navegador(quebra).abrir_site("wikipedia.org")


def test_padrao_usa_webbrowser_em_aba_nova(monkeypatch):
    chamadas = []
    monkeypatch.setattr("webbrowser.open", lambda url, new=0: chamadas.append((url, new)) or True)
    Navegador().abrir_site("wikipedia.org")
    assert chamadas == [("https://wikipedia.org", 2)]
