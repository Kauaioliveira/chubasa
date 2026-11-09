from __future__ import annotations

import pytest

from chubasa.tools.navegador import (
    ErroNavegador,
    Navegador,
    abridor,
    localizar_navegador,
    normalizar_url,
    url_google,
    url_youtube,
)


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


def test_padrao_fora_do_windows_usa_webbrowser_em_aba_nova(monkeypatch):
    chamadas = []
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr("webbrowser.open", lambda url, new=0: chamadas.append((url, new)) or True)
    Navegador().abrir_site("wikipedia.org")
    assert chamadas == [("https://wikipedia.org", 2)]


def test_padrao_no_windows_segue_o_navegador_padrao_do_sistema(monkeypatch):
    # O webbrowser do Python escolhe o Chrome quando ele está instalado, mesmo com o Edge como padrão.
    chamadas = []
    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.setattr("os.startfile", chamadas.append, raising=False)
    Navegador(abridor("padrao")).abrir_site("wikipedia.org")
    assert chamadas == ["https://wikipedia.org"]


def test_navegador_escolhido_abre_sempre_nele():
    iniciados = []
    abrir = abridor("edge", localizar=lambda nome: rf"C:\Edge\{nome}.exe", iniciar=iniciados.append)
    Navegador(abrir).pesquisar_no_google("tempo hoje")
    assert iniciados == [[r"C:\Edge\edge.exe", "https://www.google.com/search?q=tempo+hoje"]]


def test_navegador_escolhido_nao_instalado_da_erro_claro():
    abrir = abridor("firefox", localizar=lambda _nome: None, iniciar=lambda _args: None)
    with pytest.raises(ErroNavegador, match="não achei o firefox"):
        Navegador(abrir).abrir_site("wikipedia.org")


def test_localiza_pelo_app_paths_antes_do_path():
    assert localizar_navegador("edge", app_paths=lambda exe: rf"C:\X\{exe}") == r"C:\X\msedge.exe"


def test_navegador_invalido():
    with pytest.raises(ValueError, match="TOOLS_BROWSER"):
        abridor("opera")
