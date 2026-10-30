from __future__ import annotations

import importlib.util
import zipfile

import numpy as np
import pytest

from chubasa import audio
from chubasa.config import RAIZ


def carregar_script():
    caminho = RAIZ / "treino" / "gravar_amostras.py"
    spec = importlib.util.spec_from_file_location("gravar_amostras", caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


ga = carregar_script()


def fala_sintetica(antes_s=0.8, fala_s=0.6, depois_s=0.9, amplitude=0.3):
    """Silêncio com um chiado bem baixo, um tom no meio (a 'fala') e silêncio de novo."""
    gerador = np.random.default_rng(1)
    t = np.arange(int(fala_s * audio.TAXA)) / audio.TAXA
    partes = [
        gerador.normal(0, 0.0005, int(antes_s * audio.TAXA)),
        amplitude * np.sin(2 * np.pi * 220 * t),
        gerador.normal(0, 0.0005, int(depois_s * audio.TAXA)),
    ]
    return audio.para_int16(np.concatenate(partes))


def test_aparar_silencio_mantem_a_fala_com_margem():
    aparado = ga.aparar_silencio(fala_sintetica(), margem_s=0.15)
    duracao = len(aparado) / audio.TAXA
    assert 0.6 <= duracao <= 0.6 + 2 * 0.15 + 0.05
    assert audio.dbfs(aparado) > -20


def test_aparar_silencio_sem_fala_devolve_vazio():
    silencio = np.zeros(audio.TAXA, dtype=np.int16)
    assert ga.aparar_silencio(silencio).size == 0
    assert ga.aparar_silencio(np.zeros(10, dtype=np.int16)).size == 0


def test_ruido_constante_nao_conta_como_fala():
    ventilador = audio.para_int16(np.random.default_rng(2).normal(0, 0.03, audio.TAXA * 2))
    assert ga.aparar_silencio(ventilador).size == 0


def test_fala_sobre_ruido_e_aparada():
    gerador = np.random.default_rng(3)
    ruido = gerador.normal(0, 0.03, int(2.5 * audio.TAXA))
    inicio, fim = int(0.9 * audio.TAXA), int(1.5 * audio.TAXA)
    t = np.arange(fim - inicio) / audio.TAXA
    ruido[inicio:fim] += 0.4 * np.sin(2 * np.pi * 220 * t)
    aparado = ga.aparar_silencio(audio.para_int16(ruido), margem_s=0.1)
    assert 0.6 <= len(aparado) / audio.TAXA <= 0.6 + 2 * 0.1 + 0.05


def test_problemas_na_amostra():
    assert ga.problema_na_amostra(np.zeros(0, dtype=np.int16)) == "não ouvi fala nenhuma"
    baixinho = audio.para_int16(np.full(audio.TAXA, 0.001))
    assert "baixo" in ga.problema_na_amostra(baixinho)
    assert ga.problema_na_amostra(ga.aparar_silencio(fala_sintetica())) is None


def test_numeracao_continua(tmp_path):
    assert ga.proximo_indice(tmp_path, "hey_chubasa") == 1
    (tmp_path / "hey_chubasa_001.wav").touch()
    (tmp_path / "hey_chubasa_007.wav").touch()
    (tmp_path / "outra_coisa.wav").touch()
    assert ga.proximo_indice(tmp_path, "hey_chubasa") == 8


def test_gravar_serie_salva_e_repete_amostra_ruim(tmp_path, monkeypatch, capsys):
    gravacoes = iter([np.zeros(audio.TAXA, dtype=np.int16), fala_sintetica(), fala_sintetica()])
    monkeypatch.setattr(ga, "gravar", lambda _s, _d: next(gravacoes))
    monkeypatch.setattr("builtins.input", lambda _texto: "")

    feitas = ga.gravar_serie(tmp_path, "hey_chubasa", 2, 2.5, None, lambda _i: "hey chubasa")

    assert feitas == 2
    salvos = sorted(p.name for p in tmp_path.glob("*.wav"))
    assert salvos == ["hey_chubasa_001.wav", "hey_chubasa_002.wav"]
    dados, taxa = audio.ler_wav(tmp_path / salvos[0])
    assert taxa == 16_000 and dados.size < len(fala_sintetica())
    assert "Vamos de novo" in capsys.readouterr().out


def test_empacotar_gera_zip_com_estrutura(tmp_path):
    pasta = tmp_path / "amostras"
    audio.salvar_wav(pasta / "positivas" / "hey_chubasa_001.wav", fala_sintetica())
    audio.salvar_wav(pasta / "negativas" / "negativa_001.wav", fala_sintetica())

    pacote = ga.empacotar(pasta)

    assert pacote == tmp_path / "amostras.zip"
    with zipfile.ZipFile(pacote) as z:
        assert sorted(z.namelist()) == ["negativas/negativa_001.wav", "positivas/hey_chubasa_001.wav"]


def test_argumentos_invalidos():
    with pytest.raises(SystemExit):
        ga.main(["--positivas", "-1"])
