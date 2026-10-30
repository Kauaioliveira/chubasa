from __future__ import annotations

import asyncio
import io
import threading
import time

import numpy as np
import pytest

from chubasa import tts
from chubasa.config import ConfiguracaoInvalida, carregar_config

TAXA = 1000  # taxa de mentira: 1000 amostras = 1 s de "áudio"


class SintetizadorFalso:
    taxa = TAXA

    def __init__(self, atraso=0.0, falhar_em=()):
        self.atraso = atraso
        self.falhar_em = set(falhar_em)
        self.inicios: dict[str, float] = {}

    def sintetizar(self, texto):
        self.inicios[texto] = time.perf_counter()
        if texto in self.falhar_em:
            raise RuntimeError("voz quebrada")
        time.sleep(self.atraso)
        audio = np.full(int(len(texto) * 10), len(texto), dtype=np.float32)  # 10 ms por caractere
        return audio


class SaidaFalsa:
    """'Toca' em tempo real (blocos de 10 ms), respeitando o pedido de parada."""

    def __init__(self):
        self.tocados: list[int] = []  # tamanho do texto de cada áudio tocado
        self.interrompidos = 0
        self.fins: list[float] = []
        self.fechada = False

    def tocar(self, audio, taxa, deve_parar):
        for _ in range(0, len(audio), 10):
            if deve_parar():
                self.interrompidos += 1
                return
            time.sleep(10 / taxa)
        self.tocados.append(int(audio[0]) if len(audio) else 0)
        self.fins.append(time.perf_counter())

    def parar(self):
        pass

    def fechar(self):
        self.fechada = True


@pytest.fixture
def montar():
    criados = []

    def criar(**kwargs):
        falante = tts.Falante(SintetizadorFalso(**kwargs), SaidaFalsa())
        criados.append(falante)
        return falante, falante._sintetizador, falante._saida

    yield criar
    for falante in criados:
        falante.fechar()


def test_fala_em_ordem_e_liga_o_evento_falando(montar):
    falante, _, saida = montar()

    falante.falar("um")
    assert falante.falando.is_set()
    falante.falar("dois dois")
    falante.falar("três três três")

    assert falante.esperar(timeout=5)
    assert saida.tocados == [2, 9, 14]
    assert not falante.falando.is_set()


def test_sintetiza_a_proxima_enquanto_a_atual_toca(montar):
    falante, sintetizador, saida = montar()
    primeira = "primeira frase bem comprida para tocar um tempo"  # ~0.5 s de áudio

    falante.falar(primeira)
    falante.falar("segunda")
    falante.esperar(timeout=5)

    assert sintetizador.inicios["segunda"] < saida.fins[0]


def test_interromper_para_na_hora_e_descarta_a_fila(montar):
    falante, _, saida = montar()
    falante.falar("x" * 300)  # 3 s de áudio
    for i in range(5):
        falante.falar(f"frase {i} que não deve tocar")
    time.sleep(0.3)

    inicio = time.perf_counter()
    falante.interromper()
    assert falante.esperar(timeout=2)
    assert time.perf_counter() - inicio < 0.5
    assert not falante.falando.is_set()
    assert saida.tocados == []
    assert saida.interrompidos == 1

    falante.falar("de novo")
    assert falante.esperar(timeout=2)
    assert saida.tocados == [7]


def test_interromper_durante_a_sintese_nao_deixa_pendencia(montar):
    falante, _, saida = montar(atraso=0.3)
    falante.falar("sendo sintetizada")
    time.sleep(0.1)  # a thread de síntese já pegou o texto
    falante.interromper()

    assert falante.esperar(timeout=2)
    assert not falante.falando.is_set()
    time.sleep(0.4)
    assert saida.tocados == []


def test_erro_numa_frase_nao_para_as_outras(montar, caplog):
    falante, _, saida = montar(falhar_em={"quebrada"})
    falante.falar("quebrada")
    falante.falar("inteira")

    assert falante.esperar(timeout=3)
    assert saida.tocados == [7]
    assert "não consegui sintetizar" in caplog.text


def test_texto_vazio_e_ignorado(montar):
    falante, _, _ = montar()
    falante.falar("   ")
    assert not falante.falando.is_set()


def test_falar_stream_consome_o_gerador_e_espera(montar):
    falante, _, saida = montar()

    async def frases():
        for frase in ("Primeira frase.", "Segunda frase."):
            await asyncio.sleep(0.05)
            yield frase

    asyncio.run(falante.falar_stream(frases()))

    assert saida.tocados == [15, 14]
    assert not falante.falando.is_set()


def test_evento_falando_visivel_de_outra_thread(montar):
    falante, _, _ = montar()
    vistos = []
    falante.falar("x" * 50)
    leitor = threading.Thread(target=lambda: vistos.append(falante.falando.wait(timeout=1)))
    leitor.start()
    leitor.join()
    assert vistos == [True]


def test_fechar_fecha_a_saida(montar):
    falante, _, saida = montar()
    falante.fechar()
    assert saida.fechada


def test_nome_de_voz_invalido(tmp_path):
    with pytest.raises(tts.ErroTTS, match="TTS_VOICE"):
        tts.baixar_voz_piper("voz-sem-formato", tmp_path)


def test_voz_ja_baixada_nao_baixa_de_novo(tmp_path, monkeypatch):
    (tmp_path / "pt_BR-faber-medium.onnx").write_bytes(b"x")
    (tmp_path / "pt_BR-faber-medium.onnx.json").write_text("{}")
    import piper.download_voices

    monkeypatch.setattr(piper.download_voices, "download_voice", lambda *a: pytest.fail("não devia baixar"))
    assert tts.baixar_voz_piper("pt_BR-faber-medium", tmp_path) == tmp_path / "pt_BR-faber-medium.onnx"


def test_decodifica_mp3():
    import av

    taxa = 24_000
    t = np.arange(taxa) / taxa
    sinal = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    buffer = io.BytesIO()
    with av.open(buffer, "w", format="mp3") as saida:
        stream = saida.add_stream("mp3", rate=taxa, layout="mono")
        quadro = av.AudioFrame.from_ndarray(sinal.reshape(1, -1), format="flt", layout="mono")
        quadro.sample_rate = taxa
        for pacote in stream.encode(quadro):
            saida.mux(pacote)
        for pacote in stream.encode(None):
            saida.mux(pacote)

    audio = tts.decodificar_mp3(buffer.getvalue(), taxa)

    assert audio.dtype == np.float32
    assert abs(len(audio) - taxa) < taxa * 0.1
    assert 0.1 < np.abs(audio).max() <= 1.0


def test_motor_tts_invalido(env_minimo, monkeypatch):
    monkeypatch.setenv("TTS_ENGINE", "robo")
    with pytest.raises(ConfiguracaoInvalida, match="TTS_ENGINE"):
        carregar_config()


def test_padrao_e_piper_local(env_minimo):
    cfg = carregar_config()
    assert (cfg.tts_engine, cfg.tts_voice) == ("piper", "pt_BR-faber-medium")


@pytest.mark.modelos
def test_piper_real_sintetiza_portugues():
    from chubasa.config import RAIZ

    sintetizador = tts.SintetizadorPiper(tts.baixar_voz_piper("pt_BR-faber-medium", RAIZ / "models" / "piper"))
    audio = sintetizador.sintetizar("Bom dia, tudo bem?")
    assert sintetizador.taxa == 22_050
    assert len(audio) / sintetizador.taxa > 0.5
    assert np.abs(audio).max() <= 1.0


@pytest.mark.hardware
def test_toca_de_verdade():
    from chubasa.config import RAIZ

    sintetizador = tts.SintetizadorPiper(tts.baixar_voz_piper("pt_BR-faber-medium", RAIZ / "models" / "piper"))
    falante = tts.Falante(sintetizador)
    falante.falar("Teste de som.")
    assert falante.esperar(timeout=10)
    falante.fechar()
