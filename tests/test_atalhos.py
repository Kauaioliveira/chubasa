from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from chubasa.tools import Ferramentas
from chubasa.tools.atalhos import (
    Atalho,
    AtalhosDoMenuIniciar,
    ProgramaAmbiguo,
    chave,
    pastas_do_menu_iniciar,
    varrer,
)
from chubasa.tools.programas import ErroPrograma, ProgramasPermitidos

NOMES = ["Spotify", "Microsoft Word", "Word Mobile Viewer", "Paint", "Paint 3D", "Visual Studio Code",
         "Uninstall Spotify", "Desinstalar Steam", "Steam", "Calculadora Científica", "Bloco de Notas++",
         "Adobe Photoshop 2025", "OBS Studio", "Roblox Studio", "Discord", "Disk Cleanup",
         "Registry Editor", "RecoveryDrive", "Limpeza de Disco", "dfrgui"]


@pytest.fixture
def menu(tmp_path):
    comum = tmp_path / "ProgramData" / "Microsoft" / "Windows" / "Start Menu" / "Programs"
    usuario = tmp_path / "AppData" / "Microsoft" / "Windows" / "Start Menu" / "Programs"
    for i, nome in enumerate(NOMES):
        pasta = (comum if i % 2 else usuario) / ("Subpasta" if i % 3 == 0 else "")
        pasta.mkdir(parents=True, exist_ok=True)
        (pasta / f"{nome}.lnk").write_bytes(b"atalho")
    (usuario / "Spotify.lnk").write_bytes(b"duplicado")  # mesmo nome nas duas pastas
    (usuario / "Site.url").write_text("[InternetShortcut]")  # não é .lnk: ignorado
    pastas = pastas_do_menu_iniciar({"ProgramData": str(tmp_path / "ProgramData"),
                                     "APPDATA": str(tmp_path / "AppData")})
    return AtalhosDoMenuIniciar(lambda: varrer(pastas))


def test_chave_normaliza():
    assert chave("Calculadora Científica") == "calculadora cientifica"
    assert chave("  Microsoft® Word (x64)  ") == "microsoft word x64"


def test_varredura_ignora_desinstaladores_e_nao_lnk(menu):
    nomes = [a.nome for a in menu.todos()]
    assert "Uninstall Spotify" not in nomes and "Desinstalar Steam" not in nomes
    assert not {"Disk Cleanup", "Registry Editor", "RecoveryDrive", "Limpeza de Disco", "dfrgui"} & set(nomes)
    assert "Site" not in nomes
    assert nomes.count("Spotify") == 1
    assert all(a.caminho.suffix == ".lnk" for a in menu.todos())


@pytest.mark.parametrize("pedido, esperado", [
    ("spotify", "Spotify"), ("SPOTIFY", "Spotify"), ("calculadora cientifica", "Calculadora Científica"),
    ("visual studio code", "Visual Studio Code"), ("steam", "Steam"), ("photoshop", "Adobe Photoshop 2025"),
    ("obs", "OBS Studio"), ("discordi", "Discord"), ("spotfy", "Spotify"),
])
def test_procura_pelo_nome(menu, pedido, esperado):
    assert menu.procurar(pedido).nome == esperado


@pytest.mark.parametrize("pedido, opcoes", [
    ("word", ["Microsoft Word", "Word Mobile Viewer"]),
    ("studio", ["OBS Studio", "Roblox Studio", "Visual Studio Code"]),
])
def test_ambiguidade_devolve_opcoes(menu, pedido, opcoes):
    with pytest.raises(ProgramaAmbiguo) as erro:
        menu.procurar(pedido)
    assert sorted(erro.value.opcoes) == sorted(opcoes)


def test_paint_exato_vence_paint_3d(menu):
    assert menu.procurar("paint").nome == "Paint"


@pytest.mark.parametrize("pedido", ["minecraft", "", "uninstall spotify", "desinstalar steam"])
def test_nao_acha(menu, pedido):
    assert menu.procurar(pedido) is None


def test_cache_e_atualizado_depois_de_5_minutos():
    agora = [0.0]
    chamadas = []

    def listar():
        chamadas.append(1)
        return [Atalho("Spotify", Path("x.lnk"))]

    atalhos = AtalhosDoMenuIniciar(listar, relogio=lambda: agora[0])
    atalhos.todos()
    atalhos.todos()
    agora[0] = 301
    atalhos.todos()
    assert len(chamadas) == 2


def test_lista_do_env_vem_antes_do_menu(menu):
    abertos = []
    programas = ProgramasPermitidos({"calculadora": "calc.exe"}, abertos.append, menu)
    assert programas.abrir("calculadora") == "calculadora"
    assert programas.abrir("spotify") == "Spotify"
    assert abertos[0] == "calc.exe"
    assert abertos[1].endswith("Spotify.lnk") and Path(abertos[1]).is_file()  # caminho veio da varredura


def test_modelo_nunca_executa_caminho(menu):
    abertos = []
    programas = ProgramasPermitidos({}, abertos.append, menu)
    for pedido in (r"C:\Windows\System32\cmd.exe", "cmd.exe /c del *", r"..\..\Spotify.lnk"):
        try:
            programas.abrir(pedido)
        except (ErroPrograma, ProgramaAmbiguo):
            pass
    assert all(Path(a).suffix == ".lnk" and "Start Menu" in a for a in abertos)


def test_ferramenta_pergunta_quando_ambiguo(menu):
    abertos = []
    f = Ferramentas(lambda _t: None, ProgramasPermitidos({}, abertos.append, menu))
    abrir = next(d for d in f.definicoes if d.name == "abrir_programa")

    ambiguo = asyncio.run(abrir.handler({"nome": "word"}))
    achou = asyncio.run(abrir.handler({"nome": "spotify"}))
    nada = asyncio.run(abrir.handler({"nome": "minecraft"}))
    f.fechar()

    assert "Microsoft Word" in ambiguo["content"][0]["text"] and "Pergunte" in ambiguo["content"][0]["text"]
    assert not ambiguo.get("is_error")
    assert achou["content"][0]["text"] == "Abri Spotify."
    assert nada.get("is_error") and "Menu Iniciar" in nada["content"][0]["text"]
    assert len(abertos) == 1
    assert "Menu Iniciar" in abrir.description


def test_pasta_comum_tem_alternativas():
    esperado = Path(r"D:\Dados") / "Microsoft" / "Windows" / "Start Menu" / "Programs"
    assert pastas_do_menu_iniciar({"ALLUSERSPROFILE": r"D:\Dados"})[0] == esperado
    assert pastas_do_menu_iniciar({"SystemDrive": "C:"})[0].parts[:2] == ("C:\\", "ProgramData")
    assert len(pastas_do_menu_iniciar({"ProgramData": "X", "APPDATA": "Y"})) == 2
