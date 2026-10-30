from __future__ import annotations

import threading

import numpy as np
import pytest

from chubasa import audio
from sinais import fala_sintetica, seno

DISPOSITIVOS = [
    {"name": "Mapeador de som da Microsoft - Input", "hostapi": 0, "max_input_channels": 2,
     "max_output_channels": 0, "default_samplerate": 44100.0},
    {"name": "Microfone (Realtek Audio)", "hostapi": 0, "max_input_channels": 2,
     "max_output_channels": 0, "default_samplerate": 44100.0},
    {"name": "Alto-falantes (Realtek Audio)", "hostapi": 0, "max_input_channels": 0,
     "max_output_channels": 2, "default_samplerate": 44100.0},
    {"name": "Microfone (Realtek Audio)", "hostapi": 1, "max_input_channels": 2,
     "max_output_channels": 0, "default_samplerate": 48000.0},
]
APIS = [{"name": "MME"}, {"name": "Windows WASAPI"}]


class StreamFalso:
    """Imita sd.InputStream: ao iniciar, uma thread entrega frames de um sinal sintético."""

    def __init__(self, sinal: np.ndarray, **kwargs):
        self.kwargs = kwargs
        self.sinal = sinal
        self._thread: threading.Thread | None = None
        self.fechado = False

    def start(self):
        def entregar():
            passo = self.kwargs["blocksize"]
            for inicio in range(0, len(self.sinal) - passo + 1, passo):
                if self.fechado:
                    return
                bloco = self.sinal[inicio:inicio + passo].reshape(-1, 1)
                self.kwargs["callback"](bloco, passo, None, None)

        self._thread = threading.Thread(target=entregar, daemon=True)
        self._thread.start()

    def stop(self):
        self.fechado = True

    def close(self):
        self.fechado = True


@pytest.fixture
def dispositivos_falsos(monkeypatch):
    monkeypatch.setattr(audio.sd, "query_devices", lambda: DISPOSITIVOS)
    monkeypatch.setattr(audio.sd, "query_hostapis", lambda: APIS)


@pytest.fixture
def microfone_falso(monkeypatch):
    criados: list[StreamFalso] = []

    def instalar(sinal: np.ndarray):
        def fabrica(**kwargs):
            stream = StreamFalso(sinal, **kwargs)
            criados.append(stream)
            return stream

        monkeypatch.setattr(audio.sd, "InputStream", fabrica)
        return criados

    return instalar


def test_lista_so_dispositivos_de_entrada(dispositivos_falsos):
    entradas = audio.listar_dispositivos_entrada()
    assert [e["indice"] for e in entradas] == [0, 1, 3]
    assert entradas[2]["api"] == "Windows WASAPI"


@pytest.mark.parametrize("valor, esperado", [(None, None), ("", None), ("  ", None), ("1", 1), (3, 3)])
def test_resolve_vazio_e_numero(dispositivos_falsos, valor, esperado):
    assert audio.resolver_dispositivo(valor) == esperado


def test_resolve_por_parte_do_nome_pega_o_primeiro(dispositivos_falsos):
    assert audio.resolver_dispositivo("realtek") == 1


def test_numero_que_nao_e_entrada_da_erro(dispositivos_falsos):
    with pytest.raises(audio.ErroDeAudio, match="dispositivos"):
        audio.resolver_dispositivo("2")


def test_nome_inexistente_da_erro(dispositivos_falsos):
    with pytest.raises(audio.ErroDeAudio, match="headset"):
        audio.resolver_dispositivo("headset")


def test_microfone_abre_com_formato_certo_e_entrega_frames(microfone_falso):
    sinal = seno(0.4)
    criados = microfone_falso(sinal)

    with audio.Microfone(dispositivo=7) as mic:
        frames = [mic.ler(timeout=2) for _ in range(5)]

    kwargs = criados[0].kwargs
    assert kwargs["samplerate"] == 16_000
    assert kwargs["channels"] == 1
    assert kwargs["dtype"] == "int16"
    assert kwargs["blocksize"] == 1280
    assert kwargs["device"] == 7
    assert all(f.shape == (1280,) and f.dtype == np.int16 for f in frames)
    np.testing.assert_array_equal(np.concatenate(frames), sinal[:6400])
    assert criados[0].fechado


def test_fila_cheia_descarta_o_frame_mais_velho():
    mic = audio.Microfone(max_frames_na_fila=2)
    for valor in (1, 2, 3):
        mic._ao_receber(np.full((1280, 1), valor, dtype=np.int16), 1280, None, None)

    assert mic.frames_descartados == 1
    assert mic.ler()[0] == 2
    assert mic.ler()[0] == 3


def test_ler_sem_audio_da_erro_claro():
    mic = audio.Microfone()
    with pytest.raises(audio.ErroDeAudio, match="não entregou áudio"):
        mic.ler(timeout=0.05)


def test_erro_do_portaudio_vira_mensagem_util(monkeypatch):
    def falha(**_kwargs):
        raise audio.sd.PortAudioError("Invalid device")

    monkeypatch.setattr(audio.sd, "InputStream", falha)
    with pytest.raises(audio.ErroDeAudio, match="MIC_DEVICE"):
        audio.Microfone().abrir()


def test_gravar_devolve_a_duracao_exata(microfone_falso):
    microfone_falso(seno(1.2))
    gravado = audio.gravar(1.0)
    assert len(gravado) == 16_000
    assert gravado.dtype == np.int16


def test_wav_ida_e_volta(tmp_path):
    sinal = seno(0.5)
    arquivo = tmp_path / "sub" / "teste.wav"
    audio.salvar_wav(arquivo, sinal)

    lido, taxa = audio.ler_wav(arquivo)
    assert taxa == 16_000
    np.testing.assert_array_equal(lido, sinal)


def test_rms_e_dbfs():
    silencio = np.zeros(1280, dtype=np.int16)
    assert audio.rms(silencio) == 0.0
    assert audio.dbfs(silencio) == -120.0
    assert audio.rms(np.zeros(0, dtype=np.int16)) == 0.0
    # Seno de amplitude 0.5 tem RMS 0.5/sqrt(2) ~ 0.354, cerca de -9 dBFS
    assert audio.rms(seno(1.0)) == pytest.approx(0.3535, abs=1e-3)
    assert audio.dbfs(seno(1.0)) == pytest.approx(-9.03, abs=0.05)


def test_conversoes_float_int16():
    assert audio.para_int16(np.array([2.0, -2.0, 0.0])).tolist() == [32767, -32767, 0]
    assert audio.para_float32(np.array([16384], dtype=np.int16))[0] == pytest.approx(0.5)


def test_barra_de_volume():
    assert audio.barra_de_volume(np.zeros(1280, dtype=np.int16), largura=10).startswith("[..........]")
    cheio = np.full(1280, 32767, dtype=np.int16)
    assert audio.barra_de_volume(cheio, largura=10).startswith("[##########]")


def test_comando_gravar_teste_salva_arquivo(microfone_falso, tmp_path, capsys):
    microfone_falso(seno(1.2))
    destino = tmp_path / "teste.wav"

    assert audio.comando_gravar_teste(None, destino, segundos=1.0) == 0
    assert destino.exists()
    assert "Salvo em" in capsys.readouterr().out


def test_comando_gravar_teste_avisa_se_mudo(microfone_falso, tmp_path, capsys):
    microfone_falso(np.zeros(20_000, dtype=np.int16))
    assert audio.comando_gravar_teste(None, tmp_path / "x.wav", segundos=1.0) == 1
    assert "mudo" in capsys.readouterr().out


def test_aparar_silencio_mantem_a_fala_com_margem():
    aparado = audio.aparar_silencio(fala_sintetica(), margem_s=0.15)
    duracao = len(aparado) / audio.TAXA
    assert 0.6 <= duracao <= 0.6 + 2 * 0.15 + 0.05
    assert audio.dbfs(aparado) > -20


def test_aparar_silencio_sem_fala_devolve_vazio():
    silencio = np.zeros(audio.TAXA, dtype=np.int16)
    assert audio.aparar_silencio(silencio).size == 0
    assert audio.aparar_silencio(np.zeros(10, dtype=np.int16)).size == 0


def test_ruido_constante_nao_conta_como_fala():
    ventilador = audio.para_int16(np.random.default_rng(2).normal(0, 0.03, audio.TAXA * 2))
    assert audio.aparar_silencio(ventilador).size == 0


def test_fala_sobre_ruido_e_aparada():
    gerador = np.random.default_rng(3)
    ruido = gerador.normal(0, 0.03, int(2.5 * audio.TAXA))
    inicio, fim = int(0.9 * audio.TAXA), int(1.5 * audio.TAXA)
    t = np.arange(fim - inicio) / audio.TAXA
    ruido[inicio:fim] += 0.4 * np.sin(2 * np.pi * 220 * t)
    aparado = audio.aparar_silencio(audio.para_int16(ruido), margem_s=0.1)
    assert 0.6 <= len(aparado) / audio.TAXA <= 0.6 + 2 * 0.1 + 0.05


@pytest.mark.hardware
def test_microfone_real_captura_algo():
    gravado = audio.gravar(1.0)
    assert len(gravado) == 16_000
    assert audio.rms(gravado) > 0
