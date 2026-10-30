from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from chubasa import audio, stt
from chubasa.config import ConfiguracaoInvalida, carregar_config
from sinais import seno


class WhisperFalso:
    def __init__(self, segmentos):
        self.segmentos = segmentos
        self.chamadas = []

    def transcribe(self, amostras, **kwargs):
        self.chamadas.append((amostras, kwargs))
        return iter(self.segmentos), SimpleNamespace(language=kwargs["language"])


def seg(texto, no_speech=0.01):
    return SimpleNamespace(text=texto, no_speech_prob=no_speech)


def config(**extra):
    base = dict(language="pt", stt_beam_size=1, stt_model="small", stt_device="cpu",
                stt_compute_type="int8", stt_cpu_threads=0)
    base.update(extra)
    return SimpleNamespace(**base)


def test_transcreve_em_portugues_e_junta_segmentos(caplog):
    modelo = WhisperFalso([seg(" Qual vai ser"), seg(" a previsão do tempo? ")])
    transcritor = stt.Transcritor(modelo, "pt", beam_size=1)

    with caplog.at_level(logging.INFO):
        resultado = transcritor.transcrever(seno(2.0))

    assert resultado.texto == "Qual vai ser a previsão do tempo?"
    assert resultado.segundos_audio == pytest.approx(2.0)
    amostras, kwargs = modelo.chamadas[0]
    assert amostras.dtype == np.float32 and np.abs(amostras).max() <= 1.0
    assert kwargs["language"] == "pt"
    assert kwargs["beam_size"] == 1
    assert kwargs["condition_on_previous_text"] is False
    assert "transcrição:" in caplog.text


def test_descarta_segmento_que_provavelmente_nao_e_fala():
    modelo = WhisperFalso([seg("Liga a luz."), seg("Legendas pela comunidade.", no_speech=0.9)])
    assert stt.Transcritor(modelo, "pt").transcrever(seno(1.0)).texto == "Liga a luz."


def test_audio_sem_texto_devolve_vazio():
    assert stt.Transcritor(WhisperFalso([]), "pt").transcrever(np.zeros(1600, np.int16)).texto == ""


def test_criar_transcritor_carrega_uma_vez_com_config():
    carregados = []

    def carregar(cfg):
        carregados.append(cfg)
        return WhisperFalso([seg("oi")])

    transcritor = stt.criar_transcritor(config(language="pt", stt_beam_size=5), carregar=carregar)
    transcritor.transcrever(seno(0.5))
    transcritor.transcrever(seno(0.5))

    assert len(carregados) == 1
    assert transcritor.beam_size == 5


def test_stt_device_invalido(env_minimo, monkeypatch):
    monkeypatch.setenv("STT_DEVICE", "tpu")
    with pytest.raises(ConfiguracaoInvalida, match="STT_DEVICE"):
        carregar_config()


def test_padroes_do_plano(env_minimo):
    cfg = carregar_config()
    assert (cfg.stt_model, cfg.stt_device, cfg.stt_compute_type) == ("small", "cpu", "int8")


def test_comando_transcrever_arquivo(tmp_path, capsys):
    arquivo = tmp_path / "frase.wav"
    audio.salvar_wav(arquivo, seno(1.0))

    codigo = stt.comando_transcrever(config(), None, arquivo,
                                     criar=lambda _c: stt.Transcritor(WhisperFalso([seg("olá")]), "pt"))

    assert codigo == 0
    assert "olá" in capsys.readouterr().out


def test_comando_transcrever_recusa_taxa_errada(tmp_path, capsys):
    arquivo = tmp_path / "frase.wav"
    audio.salvar_wav(arquivo, seno(1.0), taxa=44_100)
    assert stt.comando_transcrever(config(), None, arquivo, criar=lambda _c: pytest.fail("não devia carregar")) == 2


@pytest.mark.modelos
def test_whisper_real_carrega_e_roda_em_silencio():
    from chubasa.config import RAIZ

    cfg = config(stt_model="tiny", models_dir=RAIZ / "models")
    transcritor = stt.criar_transcritor(cfg)
    assert isinstance(transcritor.transcrever(np.zeros(16000, np.int16)).texto, str)
