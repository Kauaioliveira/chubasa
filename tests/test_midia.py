from __future__ import annotations

import pytest

from chubasa.tools import midia
from chubasa.tools.midia import ErroMidia, TeclasDeMidia

BAIXO = midia.KEYEVENTF_EXTENDEDKEY
CIMA = midia.KEYEVENTF_EXTENDEDKEY | midia.KEYEVENTF_KEYUP


def teclas():
    enviadas = []
    return TeclasDeMidia(lambda vk, flags: enviadas.append((vk, flags)), "win32", pausa_s=0), enviadas


@pytest.mark.parametrize("acao, vk", [("tocar_pausar", 0xB3), ("proxima", 0xB0), ("anterior", 0xB1),
                                      ("parar", 0xB2), (" Proxima ", 0xB0)])
def test_midia_aperta_e_solta_a_tecla_certa(acao, vk):
    t, enviadas = teclas()
    t.midia(acao)
    assert enviadas == [(vk, BAIXO), (vk, CIMA)]


def test_volume_por_passos():
    t, enviadas = teclas()
    assert t.volume("aumentar", 3) == 3
    assert enviadas == [(0xAF, BAIXO), (0xAF, CIMA)] * 3


def test_volume_limita_passos():
    t, enviadas = teclas()
    assert t.volume("diminuir", 999) == midia.MAXIMO_PASSOS
    assert t.volume("diminuir", 0) == 1
    assert {vk for vk, _ in enviadas} == {0xAE}


def test_mudo_e_uma_tecla_so():
    t, enviadas = teclas()
    assert t.volume("mudo", 10) == 0
    assert enviadas == [(0xAD, BAIXO), (0xAD, CIMA)]


@pytest.mark.parametrize("chamar", [lambda t: t.midia("pular"), lambda t: t.volume("berrar")])
def test_acao_desconhecida(chamar):
    t, enviadas = teclas()
    with pytest.raises(ErroMidia, match="desconhecida"):
        chamar(t)
    assert enviadas == []


def test_fora_do_windows_nao_suportado():
    enviadas = []
    t = TeclasDeMidia(lambda vk, f: enviadas.append(vk), "linux")
    with pytest.raises(ErroMidia, match="Windows"):
        t.volume("aumentar", 1)
    with pytest.raises(ErroMidia, match="Windows"):
        t.midia("tocar_pausar")
    assert enviadas == []
