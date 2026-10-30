"""Texto para fala: Piper (local, voz pt-BR) por padrão, Edge TTS opcional.

`Falante` fala frase por frase: enquanto uma frase toca, a próxima já está sendo sintetizada.
O evento `falando` fica ligado do primeiro texto recebido até o fim do áudio (o loop usa isso para
não ouvir a própria voz) e `interromper()` corta a fala na hora e descarta o que estava na fila.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import shutil
import tempfile
import threading
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)

BLOCO_REPRODUCAO_S = 0.05  # de quanto em quanto tempo a reprodução confere se foi interrompida


class ErroTTS(Exception):
    """Voz indisponível ou falha ao sintetizar."""


class Sintetizador(Protocol):
    taxa: int

    def sintetizar(self, texto: str) -> np.ndarray:
        """Devolve áudio mono float32 em [-1, 1] na taxa `taxa`."""
        ...


class Saida(Protocol):
    def tocar(self, audio: np.ndarray, taxa: int, deve_parar: Callable[[], bool]) -> None: ...
    def parar(self) -> None: ...
    def fechar(self) -> None: ...


# ---------------------------------------------------------------- Piper

def baixar_voz_piper(voz: str, pasta: Path) -> Path:
    """Baixa a voz (~60 MB) uma vez. Baixa numa pasta temporária para não deixar arquivo pela metade."""
    modelo = pasta / f"{voz}.onnx"
    if modelo.is_file() and modelo.with_suffix(".onnx.json").is_file():
        return modelo
    from piper.download_voices import download_voice

    log.info("Baixando a voz %s para %s (só na primeira vez)", voz, pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=pasta) as temporaria:
        try:
            download_voice(voz, Path(temporaria))
        except ValueError as erro:
            raise ErroTTS(f"Nome de voz inválido em TTS_VOICE: {erro}") from erro
        except OSError as erro:  # urllib: sem rede, 404
            raise ErroTTS(
                f"Não consegui baixar a voz {voz} ({erro}). Confira a internet e o nome em TTS_VOICE "
                "(lista: https://huggingface.co/rhasspy/piper-voices/tree/main/pt/pt_BR)."
            ) from erro
        for arquivo in Path(temporaria).iterdir():
            shutil.move(str(arquivo), pasta / arquivo.name)
    return modelo


def sessao_onnx_sem_cache_por_formato(modelo: Path):
    """Sessão do onnxruntime sem 'memory pattern'.

    Com o padrão, o onnxruntime guarda um plano de memória para cada tamanho de entrada; como cada
    frase tem um tamanho, a memória do assistente crescia sem parar (medido: 177 -> 342 MB em 200
    frases). Sem ele fica estável (~150 MB) e a síntese fica só ~10% mais lenta.
    """
    import onnxruntime as ort

    opcoes = ort.SessionOptions()
    opcoes.enable_mem_pattern = False
    return ort.InferenceSession(str(modelo), sess_options=opcoes, providers=["CPUExecutionProvider"])


class SintetizadorPiper:
    def __init__(self, modelo: Path):
        from piper import PiperVoice

        try:
            self._voz = PiperVoice.load(modelo)
            self._voz.session = sessao_onnx_sem_cache_por_formato(modelo)
        except Exception as erro:  # onnxruntime/json inválido
            raise ErroTTS(f"Não consegui carregar a voz {modelo}: {erro}. Apague o arquivo e rode de novo.") from erro
        self.taxa = self._voz.config.sample_rate
        self.sintetizar("Oi.")  # a primeira síntese é lenta; melhor pagar isso na carga

    def sintetizar(self, texto: str) -> np.ndarray:
        pedacos = [c.audio_float_array for c in self._voz.synthesize(texto)]
        if not pedacos:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(pedacos).astype(np.float32)


# ---------------------------------------------------------------- Edge TTS (opcional, precisa de internet)

class SintetizadorEdge:
    taxa = 24_000

    def __init__(self, voz: str):
        try:
            import edge_tts  # noqa: F401
        except ImportError as erro:
            raise ErroTTS("TTS_ENGINE=edge precisa do grupo 'edge': python -m pip install -e \".[edge]\"") from erro
        self.voz = voz

    def sintetizar(self, texto: str) -> np.ndarray:
        import edge_tts

        async def baixar() -> bytes:
            partes = []
            async for item in edge_tts.Communicate(texto, self.voz).stream():
                if item["type"] == "audio":
                    partes.append(item["data"])
            return b"".join(partes)

        try:
            mp3 = asyncio.run(baixar())  # roda na thread de síntese, que não tem loop próprio
        except Exception as erro:  # aiohttp/serviço fora do ar
            raise ErroTTS(f"Edge TTS falhou ({erro}); confira a internet ou use TTS_ENGINE=piper.") from erro
        return decodificar_mp3(mp3, self.taxa)


def decodificar_mp3(dados: bytes, taxa: int) -> np.ndarray:
    import io

    import av

    amostras = []
    with av.open(io.BytesIO(dados)) as entrada:
        conversor = av.AudioResampler(format="flt", layout="mono", rate=taxa)
        for quadro in entrada.decode(audio=0):
            for convertido in conversor.resample(quadro):
                amostras.append(convertido.to_ndarray().reshape(-1))
        for convertido in conversor.resample(None):
            amostras.append(convertido.to_ndarray().reshape(-1))
    return np.concatenate(amostras).astype(np.float32) if amostras else np.zeros(0, dtype=np.float32)


def criar_sintetizador(config) -> Sintetizador:
    if config.tts_engine == "edge":
        return SintetizadorEdge(config.tts_edge_voice)
    inicio = time.perf_counter()
    sintetizador = SintetizadorPiper(baixar_voz_piper(config.tts_voice, config.models_dir / "piper"))
    log.info("Voz %s pronta em %.1f s", config.tts_voice, time.perf_counter() - inicio)
    return sintetizador


# ---------------------------------------------------------------- saída de áudio

class SaidaSounddevice:
    """Toca em blocos curtos para conseguir parar no meio de uma frase."""

    def __init__(self):
        self._stream = None
        self._taxa: int | None = None

    def tocar(self, audio: np.ndarray, taxa: int, deve_parar: Callable[[], bool]) -> None:
        import sounddevice as sd

        if self._stream is None or self._taxa != taxa:
            self.fechar()
            self._stream = sd.OutputStream(samplerate=taxa, channels=1, dtype="float32")
            self._taxa = taxa
        if not self._stream.active:
            self._stream.start()
        passo = max(1, int(BLOCO_REPRODUCAO_S * taxa))
        for inicio in range(0, len(audio), passo):
            if deve_parar():
                self.parar()
                return
            self._stream.write(audio[inicio:inicio + passo].reshape(-1, 1))

    def parar(self) -> None:
        if self._stream is not None and self._stream.active:
            self._stream.abort()  # descarta o que está no buffer da placa

    def fechar(self) -> None:
        if self._stream is not None:
            self._stream.abort()
            self._stream.close()
            self._stream = None


# ---------------------------------------------------------------- falante

class Falante:
    def __init__(self, sintetizador: Sintetizador, saida: Saida | None = None):
        self._sintetizador = sintetizador
        self._saida = saida if saida is not None else SaidaSounddevice()
        self.falando = threading.Event()
        self._textos: queue.Queue = queue.Queue()
        self._audios: queue.Queue = queue.Queue(maxsize=2)  # sintetiza no máximo 2 frases à frente
        self._trava = threading.Lock()
        self._geracao = 0
        self._pendentes = 0
        self._ocioso = threading.Condition(self._trava)
        self._fechado = False
        self._threads = [
            threading.Thread(target=self._laco_sintese, name="tts-sintese", daemon=True),
            threading.Thread(target=self._laco_reproducao, name="tts-reproducao", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    # -- interface

    def falar(self, texto: str) -> None:
        texto = texto.strip()
        if not texto:
            return
        with self._trava:
            self._pendentes += 1
            self.falando.set()
            geracao = self._geracao
        self._textos.put((geracao, texto))

    async def falar_stream(self, frases: AsyncIterator[str]) -> None:
        """Fala cada frase assim que ela chega e espera terminar (ou ser interrompida)."""
        async for frase in frases:
            self.falar(frase)
        await asyncio.to_thread(self.esperar)

    def esperar(self, timeout: float | None = None) -> bool:
        with self._ocioso:
            return self._ocioso.wait_for(lambda: self._pendentes == 0, timeout=timeout)

    def interromper(self) -> None:
        """Para a fala em até ~50 ms e descarta o que estava na fila."""
        with self._trava:
            havia_fala = self._pendentes > 0
            self._geracao += 1
            # Só o que estava nas filas é descontado aqui; o que já estava nas mãos de uma
            # thread é descontado por ela quando perceber que ficou obsoleto.
            descartados = self._esvaziar(self._textos) + self._esvaziar(self._audios)
            self._pendentes = max(0, self._pendentes - descartados)
            self._avisar_se_ocioso()
        # A thread de reprodução vê a mudança de geração e para o áudio ela mesma (sem abortar
        # o stream de outra thread no meio de um write).
        if havia_fala:
            log.info("fala interrompida")

    def fechar(self) -> None:
        self.interromper()
        self._fechado = True
        self._textos.put(None)
        for thread in self._threads:
            thread.join(timeout=2)
        self._saida.fechar()

    # -- internos

    @staticmethod
    def _esvaziar(fila: queue.Queue) -> int:
        n = 0
        while True:
            try:
                fila.get_nowait()
                n += 1
            except queue.Empty:
                return n

    def _avisar_se_ocioso(self) -> None:
        if self._pendentes == 0:
            self.falando.clear()
            self._ocioso.notify_all()

    def _concluir(self) -> None:
        with self._trava:
            self._pendentes = max(0, self._pendentes - 1)
            self._avisar_se_ocioso()

    def _atual(self, geracao: int) -> bool:
        return geracao == self._geracao and not self._fechado

    def _laco_sintese(self) -> None:
        while True:
            item = self._textos.get()
            if item is None:
                self._audios.put(None)
                return
            geracao, texto = item
            if not self._atual(geracao):
                self._concluir()
                continue
            try:
                audio = self._sintetizador.sintetizar(texto)
            except Exception as erro:  # uma frase ruim não pode matar a thread
                log.error("não consegui sintetizar %r: %s", texto, erro)
                self._concluir()
                continue
            while True:
                if not self._atual(geracao):
                    self._concluir()
                    break
                try:
                    self._audios.put((geracao, audio), timeout=0.1)
                    break
                except queue.Full:
                    continue

    def _laco_reproducao(self) -> None:
        while True:
            item = self._audios.get()
            if item is None:
                return
            geracao, audio = item
            if self._atual(geracao):
                try:
                    self._saida.tocar(audio, self._sintetizador.taxa, lambda g=geracao: not self._atual(g))
                except Exception as erro:  # placa de som sumiu etc.
                    log.error("não consegui tocar o áudio: %s", erro)
            self._concluir()


def criar_falante(config) -> Falante:
    return Falante(criar_sintetizador(config))


# ---------------------------------------------------------------- comando

def comando_falar(config, texto: str) -> int:
    falante = criar_falante(config)
    try:
        inicio = time.perf_counter()
        falante.falar(texto)
        falante.esperar()
        log.info("falado em %.2f s", time.perf_counter() - inicio)
    finally:
        falante.fechar()
    return 0
