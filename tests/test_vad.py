from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from chubasa import vad
from chubasa.audio import AMOSTRAS_POR_FRAME, TAXA


class VADFalso:
    """Probabilidade de fala = 1 se o frame tem valor > 0, senão 0."""

    def __init__(self):
        self.resets = 0

    def probabilidade(self, frame):
        return 1.0 if frame.max() > 0 else 0.0

    def reset(self):
        self.resets += 1


def frames(padrao: str):
    """'.' = silêncio, '#' = fala. Cada frame carrega o próprio índice para conferir cortes."""
    saida = []
    for i, c in enumerate(padrao):
        valor = i + 1 if c == "#" else -(i + 1)
        saida.append(np.full(AMOSTRAS_POR_FRAME, valor, dtype=np.int16))
    return saida


def indices(audio):
    return [abs(int(audio[i])) - 1 for i in range(0, len(audio), AMOSTRAS_POR_FRAME)]


PALAVRA = np.full(100, 7, dtype=np.int16)


def gravador(**kwargs):
    # Frames de 80 ms: preroll 0.16 s = 2 frames, início 0.16 s = 2 frames, silêncio 0.4 s = 5 frames,
    # cauda 0.08 s = 1 frame, timeout 0.8 s = 10 frames, máximo 1.6 s = 20 frames.
    padrao = dict(limiar=0.5, silencio_fim_s=0.4, timeout_sem_fala_s=0.8, maximo_s=1.6,
                  preroll_s=0.16, cauda_s=0.08, inicio_minimo_s=0.16)
    padrao.update(kwargs)
    return vad.GravadorComando(VADFalso(), **padrao)


def test_grava_com_preroll_e_para_no_silencio():
    entrada = frames("....####.##......." + "####")  # a fala que vem depois não entra

    resultado = gravador().gravar(iter(entrada), PALAVRA)

    assert resultado.motivo == "fim_de_fala"
    assert resultado.teve_fala
    # pré-roll (frames 2 e 3) + fala 4..10 (com a pausa curta no meio) + 1 frame de cauda (11)
    assert indices(resultado.audio_comando) == list(range(2, 12))
    np.testing.assert_array_equal(resultado.audio_palavra, PALAVRA)


def test_pausa_curta_nao_encerra():
    resultado = gravador().gravar(iter(frames("##....##....##......")), PALAVRA)
    assert indices(resultado.audio_comando)[-2:] == [13, 14]


def test_ninguem_fala_vira_sem_fala():
    resultado = gravador().gravar(iter(frames("." * 50)), PALAVRA)

    assert resultado.motivo == "sem_fala"
    assert not resultado.teve_fala
    assert resultado.audio_comando.size == 0
    np.testing.assert_array_equal(resultado.audio_palavra, PALAVRA)


def test_estalo_isolado_nao_inicia_gravacao():
    resultado = gravador().gravar(iter(frames("#.#.#.#.#.#.#.")), PALAVRA)
    assert resultado.motivo == "sem_fala"


def test_corta_no_maximo():
    resultado = gravador().gravar(iter(frames("#" * 60)), PALAVRA)
    assert resultado.motivo == "limite"
    assert len(resultado.audio_comando) == 20 * AMOSTRAS_POR_FRAME


def test_fonte_acaba_no_meio_da_fala():
    resultado = gravador().gravar(iter(frames("..###")), PALAVRA)
    assert resultado.motivo == "fim_de_fala"
    assert indices(resultado.audio_comando) == [0, 1, 2, 3, 4]


def test_reseta_o_vad_a_cada_gravacao():
    g = gravador()
    g.gravar(iter(frames("." * 20)), PALAVRA)
    g.gravar(iter(frames("." * 20)), PALAVRA)
    assert g._vad.resets == 2


def test_criar_gravador_usa_config():
    config = SimpleNamespace(vad_threshold=0.6, vad_silence_end_s=1.0, vad_no_speech_timeout_s=5.0,
                             vad_max_recording_s=30.0, vad_preroll_s=0.3)
    g = vad.criar_gravador(config, vad=VADFalso())
    assert g.limiar == 0.6
    assert g.limiar_silencio == pytest.approx(0.45)
    assert g._frames_timeout == 63  # 5 s / 80 ms, arredondado para cima


def test_download_confere_sha(tmp_path, monkeypatch):
    import requests

    class Resposta:
        content = b"modelo adulterado"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(requests, "get", lambda *_a, **_k: Resposta())
    with pytest.raises(vad.ErroVAD, match="SHA-256"):
        vad.baixar_silero(tmp_path)
    assert not (tmp_path / vad.ARQUIVO_SILERO).exists()


def test_download_sem_rede_da_erro_claro(tmp_path, monkeypatch):
    import requests

    def falha(*_a, **_k):
        raise requests.ConnectionError("sem rede")

    monkeypatch.setattr(requests, "get", falha)
    with pytest.raises(vad.ErroVAD, match="internet"):
        vad.baixar_silero(tmp_path)


def test_modelo_ja_baixado_nao_baixa_de_novo(tmp_path, monkeypatch):
    import requests

    (tmp_path / vad.ARQUIVO_SILERO).write_bytes(b"x")
    monkeypatch.setattr(requests, "get", lambda *_a, **_k: pytest.fail("não devia baixar"))
    assert vad.baixar_silero(tmp_path) == tmp_path / vad.ARQUIVO_SILERO


@pytest.mark.modelos
def test_silero_real_separa_tom_de_silencio():
    from chubasa.config import RAIZ

    modelo = vad.SileroVAD(vad.baixar_silero(RAIZ / "models" / "silero"))
    silencio = np.zeros(TAXA, dtype=np.int16)
    probs = [modelo.probabilidade(silencio[i:i + AMOSTRAS_POR_FRAME])
             for i in range(0, TAXA - AMOSTRAS_POR_FRAME + 1, AMOSTRAS_POR_FRAME)]
    assert max(probs) < 0.2
    # aceita float32 e frames de tamanho qualquer
    assert 0.0 <= modelo.probabilidade(np.zeros(300, dtype=np.float32)) <= 1.0
