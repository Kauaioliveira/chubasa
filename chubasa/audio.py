"""Captura do microfone em frames de 80 ms (16 kHz, mono, int16) e utilitários de áudio."""

from __future__ import annotations

import logging
import queue
import sys
import time
import wave
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import sounddevice as sd

TAXA = 16_000
CANAIS = 1
TIPO = "int16"
AMOSTRAS_POR_FRAME = 1280  # 80 ms a 16 kHz: o tamanho que o openWakeWord espera
DURACAO_FRAME = AMOSTRAS_POR_FRAME / TAXA

log = logging.getLogger(__name__)


class ErroDeAudio(Exception):
    """Problema com o dispositivo de áudio (não existe, não abre, não captura)."""


# ---------------------------------------------------------------- dispositivos

def listar_dispositivos_entrada() -> list[dict]:
    """Dispositivos que têm canal de entrada, com o nome da API do Windows/Linux."""
    apis = sd.query_hostapis()
    entradas = []
    for indice, disp in enumerate(sd.query_devices()):
        if disp["max_input_channels"] < 1:
            continue
        entradas.append(
            {
                "indice": indice,
                "nome": " ".join(disp["name"].split()),  # alguns drivers põem quebra de linha
                "api": apis[disp["hostapi"]]["name"],
                "canais": disp["max_input_channels"],
                "taxa_padrao": disp["default_samplerate"],
            }
        )
    return entradas


def entrada_padrao() -> int | None:
    padrao = sd.default.device[0]
    return padrao if isinstance(padrao, int) and padrao >= 0 else None


def resolver_dispositivo(valor: str | int | None) -> int | None:
    """Converte o MIC_DEVICE do .env (vazio, número ou parte do nome) no índice do dispositivo."""
    if valor is None or (isinstance(valor, str) and not valor.strip()):
        return None
    entradas = listar_dispositivos_entrada()
    texto = str(valor).strip()
    if texto.isdigit():
        indice = int(texto)
        if any(e["indice"] == indice for e in entradas):
            return indice
        raise ErroDeAudio(
            f"MIC_DEVICE={indice} não é um dispositivo de entrada. "
            "Veja a lista com: python -m chubasa dispositivos"
        )
    achados = [e for e in entradas if texto.lower() in e["nome"].lower()]
    if not achados:
        raise ErroDeAudio(
            f"Nenhum microfone com '{texto}' no nome. Veja a lista com: python -m chubasa dispositivos"
        )
    if len(achados) > 1:
        # O mesmo microfone aparece uma vez por API (MME, WASAPI...); fica o primeiro (MME no Windows,
        # que converte a taxa sozinho). Para escolher outro, use o número.
        log.info("MIC_DEVICE='%s' casa com %d dispositivos; usando %d (%s)",
                 texto, len(achados), achados[0]["indice"], achados[0]["api"])
    return achados[0]["indice"]


# ---------------------------------------------------------------- captura

class Microfone:
    """Lê o microfone numa thread do PortAudio e entrega frames de 80 ms por uma fila.

    Uso:
        with Microfone(dispositivo) as mic:
            for frame in mic.frames():
                ...
    """

    def __init__(self, dispositivo: int | None = None, max_frames_na_fila: int = 250):
        self.dispositivo = dispositivo
        self._fila: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_frames_na_fila)
        self._stream: sd.InputStream | None = None
        self.frames_descartados = 0

    def __enter__(self) -> Microfone:
        self.abrir()
        return self

    def __exit__(self, *_exc) -> None:
        self.fechar()

    def abrir(self) -> None:
        try:
            self._stream = sd.InputStream(
                samplerate=TAXA,
                channels=CANAIS,
                dtype=TIPO,
                blocksize=AMOSTRAS_POR_FRAME,
                device=self.dispositivo,
                callback=self._ao_receber,
            )
            self._stream.start()
        except sd.PortAudioError as erro:
            self._stream = None
            raise ErroDeAudio(
                f"Não consegui abrir o microfone ({erro}). Confira se ele está conectado e liberado em "
                "Configurações > Privacidade > Microfone, ou escolha outro em MIC_DEVICE."
            ) from erro
        log.debug("microfone aberto (dispositivo=%s)", self.dispositivo)

    def fechar(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            log.debug("microfone fechado")
        if self.frames_descartados:
            log.warning("%d frames descartados porque a fila encheu (consumo lento)", self.frames_descartados)

    def _ao_receber(self, dados: np.ndarray, _frames: int, _tempo, status: sd.CallbackFlags) -> None:
        # Roda na thread do PortAudio: nada de trabalho pesado aqui.
        if status:
            log.debug("status do áudio: %s", status)
        frame = dados[:, 0].copy()
        try:
            self._fila.put_nowait(frame)
        except queue.Full:
            # Melhor perder o áudio mais velho do que travar a captura.
            try:
                self._fila.get_nowait()
            except queue.Empty:
                pass
            self.frames_descartados += 1
            self._fila.put_nowait(frame)

    def ler(self, timeout: float = 1.0) -> np.ndarray:
        try:
            return self._fila.get(timeout=timeout)
        except queue.Empty:
            raise ErroDeAudio(f"O microfone não entregou áudio em {timeout:.1f} s.") from None

    def frames(self, timeout: float = 1.0) -> Iterator[np.ndarray]:
        while True:
            yield self.ler(timeout)


def gravar(segundos: float, dispositivo: int | None = None) -> np.ndarray:
    total = int(round(segundos * TAXA))
    pedacos: list[np.ndarray] = []
    with Microfone(dispositivo) as mic:
        obtido = 0
        while obtido < total:
            frame = mic.ler(timeout=2.0)
            pedacos.append(frame)
            obtido += len(frame)
    return np.concatenate(pedacos)[:total] if pedacos else np.zeros(0, dtype=np.int16)


# ---------------------------------------------------------------- utilitários

def rms(audio: np.ndarray) -> float:
    """Volume RMS normalizado entre 0 e 1."""
    if audio.size == 0:
        return 0.0
    normalizado = audio.astype(np.float64) / 32768.0
    return float(np.sqrt(np.mean(normalizado**2)))


def dbfs(audio: np.ndarray) -> float:
    valor = rms(audio)
    return 20 * np.log10(valor) if valor > 0 else -120.0


def para_float32(audio: np.ndarray) -> np.ndarray:
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    return audio.astype(np.float32)


def para_int16(audio: np.ndarray) -> np.ndarray:
    if audio.dtype == np.int16:
        return audio
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)


def salvar_wav(caminho: Path, audio: np.ndarray, taxa: int = TAXA) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(caminho), "wb") as arquivo:
        arquivo.setnchannels(1)
        arquivo.setsampwidth(2)
        arquivo.setframerate(taxa)
        arquivo.writeframes(para_int16(audio).tobytes())


def ler_wav(caminho: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(caminho), "rb") as arquivo:
        if arquivo.getsampwidth() != 2:
            raise ErroDeAudio(f"{caminho}: só WAV de 16 bits é suportado")
        taxa = arquivo.getframerate()
        canais = arquivo.getnchannels()
        dados = np.frombuffer(arquivo.readframes(arquivo.getnframes()), dtype=np.int16)
    if canais > 1:
        dados = dados.reshape(-1, canais)[:, 0]
    return dados.copy(), taxa


def barra_de_volume(audio: np.ndarray, largura: int = 40) -> str:
    """Barra de -60 dBFS (vazia) a 0 dBFS (cheia)."""
    nivel = dbfs(audio)
    cheio = int(round(max(0.0, min(1.0, (nivel + 60) / 60)) * largura))
    return f"[{'#' * cheio}{'.' * (largura - cheio)}] {nivel:6.1f} dBFS"


# ---------------------------------------------------------------- comandos

def comando_dispositivos() -> int:
    entradas = listar_dispositivos_entrada()
    padrao = entrada_padrao()
    if not entradas:
        print("Nenhum dispositivo de entrada encontrado.")
        return 1
    print("Dispositivos de entrada (use o número ou parte do nome em MIC_DEVICE):")
    for e in entradas:
        marca = "*" if e["indice"] == padrao else " "
        print(f" {marca} {e['indice']:>3}  {e['nome']}  [{e['api']}, {e['canais']} canais, "
              f"{e['taxa_padrao']:.0f} Hz]")
    print("  * = padrão do sistema")
    return 0


def comando_gravar_teste(dispositivo: int | None, caminho: Path, segundos: float = 5.0) -> int:
    print(f"Gravando {segundos:.0f} s... fale agora.")
    audio = gravar(segundos, dispositivo)
    salvar_wav(caminho, audio)
    print(f"Salvo em {caminho} ({len(audio) / TAXA:.1f} s, volume {dbfs(audio):.1f} dBFS, RMS {rms(audio):.4f})")
    if rms(audio) == 0:
        print("Atenção: o áudio veio mudo. Confira o microfone e o MIC_DEVICE.")
        return 1
    return 0


def comando_medidor(dispositivo: int | None, segundos: float | None = None) -> int:
    print("Medidor de volume (Ctrl+C para sair)")
    inicio = time.monotonic()
    try:
        with Microfone(dispositivo) as mic:
            for frame in mic.frames():
                sys.stdout.write("\r" + barra_de_volume(frame))
                sys.stdout.flush()
                if segundos is not None and time.monotonic() - inicio >= segundos:
                    break
    except KeyboardInterrupt:
        pass
    print()
    return 0
