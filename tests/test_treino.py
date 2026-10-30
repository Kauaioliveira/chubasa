from __future__ import annotations

import importlib.util
import zipfile

import numpy as np
import pytest

from chubasa import audio
from chubasa.config import RAIZ
from sinais import fala_sintetica


def carregar_script():
    caminho = RAIZ / "treino" / "gravar_amostras.py"
    spec = importlib.util.spec_from_file_location("gravar_amostras", caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


ga = carregar_script()


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
