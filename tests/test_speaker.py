from __future__ import annotations

import logging
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from chubasa import speaker
from chubasa.audio import TAXA, para_float32, para_int16
from sinais import fala_sintetica, seno


def codificador_por_frequencia(audio: np.ndarray) -> np.ndarray:
    """Falso 'modelo de voz': o embedding é o espectro grosseiro do áudio.

    Sinais com a mesma frequência dominante ficam parecidos (mesma 'voz'), frequências
    diferentes ficam distantes (outra 'voz'). Basta para testar a lógica sem baixar modelo.
    """
    espectro = np.abs(np.fft.rfft(para_float32(audio), n=TAXA))[:2000]
    return espectro.reshape(40, 50).sum(axis=1).astype(np.float32)


def voz(freq: float, segundos: float = 1.5, semente: int = 0) -> np.ndarray:
    ruido = np.random.default_rng(semente).normal(0, 0.01, int(segundos * TAXA))
    return para_int16(para_float32(seno(segundos, freq, 0.4)) + ruido)


def config(tmp_path, **extra):
    base = dict(speaker_verify=True, speaker_profile_path=tmp_path / "voz.npy",
                speaker_threshold=0.9, models_dir=tmp_path)
    base.update(extra)
    return SimpleNamespace(**base)


def test_similaridade_cosseno():
    assert speaker.similaridade_cosseno(np.array([1, 0]), np.array([2, 0])) == pytest.approx(1.0)
    assert speaker.similaridade_cosseno(np.array([1, 0]), np.array([0, 3])) == pytest.approx(0.0)
    assert speaker.similaridade_cosseno(np.array([1, 0]), np.array([-1, 0])) == pytest.approx(-1.0)


def test_vetor_nulo_da_erro():
    with pytest.raises(speaker.ErroVoz):
        speaker.normalizar(np.zeros(4))


def test_criar_perfil_e_media_normalizada():
    perfil, scores = speaker.criar_perfil([voz(200, semente=i) for i in range(5)], codificador_por_frequencia)

    assert np.linalg.norm(perfil) == pytest.approx(1.0)
    assert len(scores) == 5
    assert min(scores) > 0.95


def test_criar_perfil_sem_amostras():
    with pytest.raises(speaker.ErroVoz, match="nenhuma amostra"):
        speaker.criar_perfil([], codificador_por_frequencia)


def test_perfil_ida_e_volta(tmp_path):
    caminho = tmp_path / "sub" / "voz.npy"
    speaker.salvar_perfil(caminho, np.array([3.0, 4.0]))

    np.testing.assert_allclose(speaker.carregar_perfil(caminho), [0.6, 0.8], rtol=1e-6)
    assert speaker.carregar_perfil(tmp_path / "nao-existe.npy") is None


def test_perfil_corrompido_da_erro_acionavel(tmp_path):
    caminho = tmp_path / "voz.npy"
    caminho.write_bytes(b"isso nao e um npy")
    with pytest.raises(speaker.ErroVoz, match="cadastrar-voz"):
        speaker.carregar_perfil(caminho)


def test_sem_cadastro_aceita_qualquer_voz_e_avisa(tmp_path, caplog):
    carregou = []
    with caplog.at_level(logging.WARNING):
        verificador = speaker.criar_verificador(config(tmp_path), carregar=lambda _p: carregou.append(1))

    resultado = verificador.verificar(voz(500))

    assert resultado == speaker.ResultadoVoz(aceito=True, score=None, verificado=False)
    assert not verificador.ativo
    assert carregou == []  # nem carrega o modelo
    assert "Nenhuma voz cadastrada" in caplog.text


def test_verificacao_desligada_no_env(tmp_path, caplog):
    speaker.salvar_perfil(tmp_path / "voz.npy", np.ones(40))
    with caplog.at_level(logging.WARNING):
        verificador = speaker.criar_verificador(config(tmp_path, speaker_verify=False),
                                                carregar=lambda _p: codificador_por_frequencia)

    assert verificador.verificar(voz(500)).verificado is False
    assert "SPEAKER_VERIFY=false" in caplog.text


def test_aceita_a_voz_cadastrada_e_rejeita_outra(tmp_path, caplog):
    perfil, _ = speaker.criar_perfil([voz(200, semente=i) for i in range(5)], codificador_por_frequencia)
    speaker.salvar_perfil(tmp_path / "voz.npy", perfil)
    verificador = speaker.criar_verificador(config(tmp_path), carregar=lambda _p: codificador_por_frequencia)

    with caplog.at_level(logging.INFO):
        minha = verificador.verificar(voz(200, 1.0, semente=10), voz(200, 2.0, semente=11))
        outra = verificador.verificar(voz(900, 1.0, semente=12), voz(900, 2.0, semente=13))

    assert minha.aceito and minha.verificado and minha.score > 0.9
    assert not outra.aceito and outra.score < 0.9
    assert "voz: score=" in caplog.text and "rejeitada" in caplog.text


def test_usa_palavra_e_comando_juntos(tmp_path):
    recebidos = []

    def codificar(audio):
        recebidos.append(audio)
        return np.ones(4)

    verificador = speaker.VerificadorVoz(np.ones(4), 0.5, lambda: codificar)
    verificador.verificar(np.zeros(1000, dtype=np.int16), np.zeros(3000, dtype=np.int16))
    verificador.verificar(np.zeros(1000, dtype=np.int16))

    assert [len(a) for a in recebidos] == [4000, 1000]
    assert recebidos[0].dtype == np.float32


def test_audio_vazio_e_rejeitado():
    verificador = speaker.VerificadorVoz(np.ones(4), 0.5, lambda: (lambda _a: np.ones(4)))
    resultado = verificador.verificar(np.zeros(0, dtype=np.int16))
    assert not resultado.aceito and resultado.verificado


def test_modelo_carrega_uma_vez_so(tmp_path):
    cargas = []

    def carregar():
        cargas.append(1)
        return lambda _a: np.ones(4)

    verificador = speaker.VerificadorVoz(np.ones(4), 0.5, carregar)
    verificador.preparar()
    verificador.verificar(voz(200))
    verificador.verificar(voz(200))
    assert cargas == [1]


def test_sem_speechbrain_explica_como_instalar(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "speechbrain", None)
    monkeypatch.setitem(sys.modules, "speechbrain.inference", None)
    monkeypatch.setitem(sys.modules, "speechbrain.inference.speaker", None)
    with pytest.raises(speaker.ErroVoz, match=r"\.\[voz\]"):
        speaker.carregar_ecapa(tmp_path)


def test_cadastrar_voz_salva_perfil_e_repete_frase_ruim(tmp_path, capsys):
    gravacoes = iter([np.zeros(5 * TAXA, dtype=np.int16)] + [fala_sintetica(fala_s=1.5, semente=i) for i in range(10)])
    perguntas = []

    codigo = speaker.comando_cadastrar_voz(
        config(tmp_path), None, n_frases=10, segundos=5.0,
        carregar=lambda _p: codificador_por_frequencia,
        gravar_audio=lambda _s, _d: next(gravacoes),
        perguntar=perguntas.append,
    )

    assert codigo == 0
    assert len(perguntas) == 11  # uma repetida
    assert speaker.carregar_perfil(tmp_path / "voz.npy").shape == (40,)
    saida = capsys.readouterr().out
    assert "Vamos de novo" in saida and "Voz cadastrada" in saida
    assert not list(tmp_path.glob("*.wav"))  # nenhuma gravação fica em disco


@pytest.mark.parametrize("n", [9, 21])
def test_cadastrar_voz_exige_10_a_20_frases(tmp_path, n):
    assert speaker.comando_cadastrar_voz(config(tmp_path), None, n_frases=n,
                                         carregar=lambda _p: pytest.fail("não devia carregar")) == 2


@pytest.mark.modelos
def test_ecapa_real_gera_embedding_de_192(tmp_path):
    from chubasa.config import RAIZ

    codificar = speaker.carregar_ecapa(RAIZ / "models")
    embedding = codificar(voz(200, 2.0))
    assert embedding.shape == (192,)
    assert np.all(np.isfinite(embedding))
