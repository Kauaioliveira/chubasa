from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from chubasa import wakeword
from chubasa.audio import AMOSTRAS_POR_FRAME, TAXA


class ModeloFalso:
    """Devolve scores pré-definidos, um por frame."""

    def __init__(self, nome: str, scores: list[float]):
        self.nome = nome
        self.scores = list(scores)
        self.resets = 0
        self.frames: list[np.ndarray] = []

    def predict(self, x):
        self.frames.append(x)
        return {self.nome: self.scores.pop(0) if self.scores else 0.0}

    def reset(self):
        self.resets += 1


def frame(valor: int = 0) -> np.ndarray:
    return np.full(AMOSTRAS_POR_FRAME, valor, dtype=np.int16)


def criar(scores, limiar=0.5, cooldown=2.0, historico=2.0):
    modelo = ModeloFalso("hey_chubasa", scores)
    detector = wakeword.DetectorPalavra(modelo, "hey_chubasa", "hey chubasa", limiar, cooldown,
                                        segundos_de_historico=historico)
    return detector, modelo


def test_detecta_acima_do_limiar_e_reseta_o_modelo():
    detector, modelo = criar([0.1, 0.49, 0.8])

    assert detector.processar(frame()) is None
    assert detector.processar(frame()) is None
    deteccao = detector.processar(frame())

    assert deteccao is not None
    assert deteccao.score == pytest.approx(0.8)
    assert deteccao.modelo == "hey_chubasa"
    assert deteccao.instante == pytest.approx(3 * 0.08)
    assert modelo.resets == 1
    assert detector.ultimo_score == pytest.approx(0.8)


def test_cooldown_conta_tempo_de_audio():
    # cooldown de 0.4 s = 5 frames de 80 ms
    detector, _ = criar([0.9] * 10, cooldown=0.4)

    resultados = [detector.processar(frame()) is not None for _ in range(10)]

    assert resultados == [True, False, False, False, False, True, False, False, False, False]


def test_cooldown_zero_permite_detecoes_seguidas():
    detector, _ = criar([0.9, 0.9], cooldown=0.0)
    assert detector.processar(frame()) is not None
    assert detector.processar(frame()) is not None


def test_modelo_sem_a_chave_conta_como_zero():
    modelo = ModeloFalso("outro", [0.99])
    detector = wakeword.DetectorPalavra(modelo, "hey_chubasa", "hey chubasa")
    assert detector.processar(frame()) is None


def test_audio_recente_guarda_so_os_ultimos_segundos():
    detector, _ = criar([], historico=0.4)  # 0.4 s = 5 frames
    assert detector.audio_recente().size == 0

    for valor in range(8):
        detector.processar(frame(valor))

    recente = detector.audio_recente()
    assert len(recente) == 5 * AMOSTRAS_POR_FRAME
    assert recente[0] == 3 and recente[-1] == 7


def test_escolhe_modelo_treinado_quando_existe(tmp_path, caplog):
    treinado = tmp_path / "hey_chubasa.onnx"
    treinado.write_bytes(b"x")

    with caplog.at_level(logging.WARNING):
        arquivo, nome, frase = wakeword.escolher_modelo(treinado, tmp_path / "base", "hey chubasa")

    assert (arquivo, nome, frase) == (treinado, "hey_chubasa", "hey chubasa")
    assert caplog.text == ""


def test_sem_modelo_treinado_cai_no_hey_jarvis_com_aviso(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        arquivo, nome, frase = wakeword.escolher_modelo(tmp_path / "hey_chubasa.onnx", tmp_path, "hey chubasa")

    assert arquivo == tmp_path / "hey_jarvis_v0.1.onnx"
    assert nome == "hey_jarvis_v0.1"
    assert frase == "hey jarvis"
    assert "não encontrado" in caplog.text
    assert "treino/LEIA-ME.md" in caplog.text


def test_garantir_modelos_base_baixa_so_se_faltar(tmp_path):
    chamadas = []

    def baixar(model_names, target_directory):
        chamadas.append((model_names, target_directory))
        for nome in (*wakeword.ARQUIVOS_BASE, wakeword.ARQUIVO_RESERVA):
            (Path(target_directory) / nome).write_bytes(b"x")

    pasta = tmp_path / "openwakeword"
    wakeword.garantir_modelos_base(pasta, baixar)
    wakeword.garantir_modelos_base(pasta, baixar)

    assert chamadas == [(["hey_jarvis"], str(pasta))]


def test_falha_no_download_vira_erro_claro(tmp_path):
    def baixar(**_kwargs):
        raise ConnectionError("sem rede")

    with pytest.raises(wakeword.ErroWakeWord, match="internet"):
        wakeword.garantir_modelos_base(tmp_path, baixar)


def test_download_incompleto_vira_erro_claro(tmp_path):
    with pytest.raises(wakeword.ErroWakeWord, match="faltam"):
        wakeword.garantir_modelos_base(tmp_path, lambda **_kwargs: None)


def test_criar_detector_usa_config(tmp_path):
    pasta_base = tmp_path / "openwakeword"
    pasta_base.mkdir()
    for nome in (*wakeword.ARQUIVOS_BASE, wakeword.ARQUIVO_RESERVA):
        (pasta_base / nome).write_bytes(b"x")
    config = SimpleNamespace(models_dir=tmp_path, wake_model_path=tmp_path / "hey_chubasa.onnx",
                             wake_word="hey chubasa", wake_threshold=0.7, wake_cooldown_s=3.0)
    carregados = []

    def carregar(arquivo, base):
        carregados.append((arquivo, base))
        return ModeloFalso("hey_jarvis_v0.1", [])

    detector = wakeword.criar_detector(config, carregar=carregar)

    assert carregados == [(pasta_base / "hey_jarvis_v0.1.onnx", pasta_base)]
    assert detector.limiar == 0.7
    assert detector.cooldown_s == 3.0
    assert detector.frase == "hey jarvis"


def test_bip_curto_sem_estalo():
    bip = wakeword.gerar_bip(duracao=0.12, volume=0.3)

    assert bip.dtype == np.float32
    assert len(bip) == int(0.12 * TAXA)
    assert np.max(np.abs(bip)) <= 0.3 + 1e-6
    assert abs(bip[0]) < 1e-3 and abs(bip[-1]) < 1e-3
    # Frequência dominante perto de 880 Hz
    espectro = np.abs(np.fft.rfft(bip))
    pico_hz = np.argmax(espectro) * TAXA / len(bip)
    assert pico_hz == pytest.approx(880, abs=20)


@pytest.mark.modelos
def test_modelo_real_nao_dispara_com_silencio_nem_ruido(tmp_path):
    from chubasa.config import RAIZ

    pasta = RAIZ / "models" / "openwakeword"
    wakeword.garantir_modelos_base(pasta)
    modelo = wakeword.carregar_openwakeword(pasta / wakeword.ARQUIVO_RESERVA, pasta)
    detector = wakeword.DetectorPalavra(modelo, "hey_jarvis_v0.1", "hey jarvis")
    gerador = np.random.default_rng(0)
    ruido = (gerador.normal(0, 0.05, TAXA * 3) * 32767).astype(np.int16)
    sinal = np.concatenate([np.zeros(TAXA * 2, dtype=np.int16), ruido])

    deteccoes = [detector.processar(sinal[i:i + AMOSTRAS_POR_FRAME])
                 for i in range(0, len(sinal) - AMOSTRAS_POR_FRAME + 1, AMOSTRAS_POR_FRAME)]

    assert not any(deteccoes)
    assert detector.ultimo_score < 0.1


class MicrofoneFalso:
    """Entrega os frames de um sinal e depois simula o Ctrl+C."""

    def __init__(self, sinal: np.ndarray):
        self._frames = [sinal[i:i + AMOSTRAS_POR_FRAME]
                        for i in range(0, len(sinal) - AMOSTRAS_POR_FRAME + 1, AMOSTRAS_POR_FRAME)]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def ler(self, timeout: float = 1.0):
        if not self._frames:
            raise KeyboardInterrupt
        return self._frames.pop(0)

    def frames(self):
        while True:
            yield self.ler()


class VerificadorFalso:
    def __init__(self, ativo=True):
        self.ativo = ativo
        self.chamadas = []
        self.preparado = False

    def preparar(self):
        self.preparado = True

    def verificar(self, palavra, comando=None):
        self.chamadas.append((palavra, comando))


def test_escutar_verifica_palavra_mais_comando(monkeypatch):
    # 3 frames sem nada, detecção no 4º, depois 2 s de "comando" (25 frames) e mais 5 frames.
    sinal = np.arange(33 * AMOSTRAS_POR_FRAME, dtype=np.int64).astype(np.int16)
    detector, _ = criar([0.0, 0.0, 0.0, 0.9], cooldown=10.0)
    monkeypatch.setattr(wakeword, "criar_detector", lambda _config: detector)
    monkeypatch.setattr(wakeword, "Microfone", lambda _disp: MicrofoneFalso(sinal))
    monkeypatch.setattr(wakeword, "tocar_bip", lambda: None)
    verificador = VerificadorFalso()

    assert wakeword.comando_escutar(SimpleNamespace(), None, verificador, segundos_comando=2.0) == 0

    assert verificador.preparado
    assert len(verificador.chamadas) == 1
    palavra, comando = verificador.chamadas[0]
    np.testing.assert_array_equal(palavra, sinal[:4 * AMOSTRAS_POR_FRAME])
    np.testing.assert_array_equal(comando, sinal[4 * AMOSTRAS_POR_FRAME:29 * AMOSTRAS_POR_FRAME])


def test_escutar_sem_cadastro_nao_espera_comando(monkeypatch):
    detector, _ = criar([0.9], cooldown=10.0)
    monkeypatch.setattr(wakeword, "criar_detector", lambda _config: detector)
    monkeypatch.setattr(wakeword, "Microfone", lambda _disp: MicrofoneFalso(np.zeros(5 * AMOSTRAS_POR_FRAME, np.int16)))
    bips = []
    monkeypatch.setattr(wakeword, "tocar_bip", lambda: bips.append(1))
    verificador = VerificadorFalso(ativo=False)

    wakeword.comando_escutar(SimpleNamespace(), None, verificador)

    assert bips == [1]
    assert verificador.chamadas == []
