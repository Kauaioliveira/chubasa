"""Detecção da palavra-chave com openWakeWord (ONNX), consumindo os frames de 80 ms do microfone."""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from chubasa.audio import DURACAO_FRAME, TAXA, Microfone

# Modelo pronto usado enquanto o "hey chubasa" não for treinado
MODELO_RESERVA = "hey_jarvis"
ARQUIVO_RESERVA = "hey_jarvis_v0.1.onnx"
FRASE_RESERVA = "hey jarvis"
# Modelos de pré-processamento que todo modelo do openWakeWord precisa
ARQUIVOS_BASE = ("melspectrogram.onnx", "embedding_model.onnx")

log = logging.getLogger(__name__)


class ErroWakeWord(Exception):
    """Modelo da palavra-chave não pôde ser carregado ou baixado."""


class ModeloWakeWord(Protocol):
    def predict(self, x: np.ndarray) -> dict[str, float]: ...
    def reset(self) -> None: ...


@dataclass(frozen=True)
class Deteccao:
    modelo: str
    score: float
    instante: float  # segundos de áudio processados desde o início


class DetectorPalavra:
    """Recebe frames, devolve uma Deteccao quando o score passa do limiar, respeitando o cooldown.

    O cooldown é medido em tempo de áudio (frames processados), não no relógio: assim ele vale
    mesmo quando a fila acumula atraso ou quando o áudio vem de arquivo.

    Guarda os últimos segundos de áudio para a verificação de voz usar o trecho da palavra-chave.
    """

    def __init__(
        self,
        modelo: ModeloWakeWord,
        nome_modelo: str,
        frase: str,
        limiar: float = 0.5,
        cooldown_s: float = 2.0,
        segundos_de_historico: float = 2.0,
    ):
        self._modelo = modelo
        self.nome_modelo = nome_modelo
        self.frase = frase
        self.limiar = limiar
        self.cooldown_s = cooldown_s
        self._segundos_de_audio = 0.0
        self._historico: deque[np.ndarray] = deque(maxlen=math.ceil(segundos_de_historico / DURACAO_FRAME))
        self._ultima: float | None = None
        self.ultimo_score = 0.0

    def processar(self, frame: np.ndarray) -> Deteccao | None:
        self._historico.append(frame)
        self._segundos_de_audio += len(frame) / TAXA
        scores = self._modelo.predict(frame)
        score = float(scores.get(self.nome_modelo, 0.0))
        self.ultimo_score = score
        if score < self.limiar:
            return None

        agora = self._segundos_de_audio
        if self._ultima is not None and agora - self._ultima < self.cooldown_s:
            return None
        self._ultima = agora
        # Zera os buffers internos para a mesma fala não disparar de novo nos próximos frames.
        self._modelo.reset()
        return Deteccao(self.nome_modelo, score, agora)

    def audio_recente(self) -> np.ndarray:
        """Últimos segundos de áudio (int16), que contêm a palavra-chave logo após uma detecção."""
        if not self._historico:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(list(self._historico))


# ---------------------------------------------------------------- modelos

def garantir_modelos_base(pasta: Path, baixar: Callable[..., None] | None = None) -> None:
    """Baixa (uma vez) os modelos de pré-processamento e o modelo reserva para `pasta`."""
    necessarios = [pasta / nome for nome in (*ARQUIVOS_BASE, ARQUIVO_RESERVA)]
    if all(caminho.exists() for caminho in necessarios):
        return
    if baixar is None:
        from openwakeword.utils import download_models as baixar

    log.info("Baixando modelos do openWakeWord para %s (só na primeira vez)", pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    try:
        baixar(model_names=[MODELO_RESERVA], target_directory=str(pasta))
    except Exception as erro:  # requests/IO: qualquer falha vira mensagem acionável
        raise ErroWakeWord(
            f"Não consegui baixar os modelos do openWakeWord ({erro}). "
            "Confira a conexão com a internet e tente de novo."
        ) from erro
    faltando = [caminho.name for caminho in necessarios if not caminho.exists()]
    if faltando:
        raise ErroWakeWord(
            f"Download incompleto, faltam: {', '.join(faltando)}. Apague a pasta {pasta} e tente de novo."
        )


def escolher_modelo(caminho_desejado: Path, pasta_base: Path, frase_desejada: str) -> tuple[Path, str, str]:
    """Devolve (arquivo, nome do modelo, frase). Cai no 'hey jarvis' se o modelo treinado não existir."""
    if caminho_desejado.is_file():
        return caminho_desejado, caminho_desejado.stem, frase_desejada
    log.warning(
        "Modelo da palavra-chave não encontrado em %s. Usando o modelo pronto '%s': "
        "diga \"%s\" para testar. Para treinar '%s', veja treino/LEIA-ME.md.",
        caminho_desejado, FRASE_RESERVA, FRASE_RESERVA, frase_desejada,
    )
    return pasta_base / ARQUIVO_RESERVA, Path(ARQUIVO_RESERVA).stem, FRASE_RESERVA


def carregar_openwakeword(arquivo: Path, pasta_base: Path) -> ModeloWakeWord:
    from openwakeword.model import Model

    try:
        return Model(
            wakeword_models=[str(arquivo)],
            inference_framework="onnx",  # tflite não existe no Windows
            melspec_model_path=str(pasta_base / ARQUIVOS_BASE[0]),
            embedding_model_path=str(pasta_base / ARQUIVOS_BASE[1]),
        )
    except Exception as erro:  # onnxruntime dá erros variados para arquivo corrompido/incompatível
        raise ErroWakeWord(f"Não consegui carregar o modelo {arquivo}: {erro}") from erro


def criar_detector(config, carregar: Callable[[Path, Path], ModeloWakeWord] = carregar_openwakeword,
                   baixar: Callable[..., None] | None = None) -> DetectorPalavra:
    pasta_base = config.models_dir / "openwakeword"
    garantir_modelos_base(pasta_base, baixar)
    arquivo, nome, frase = escolher_modelo(config.wake_model_path, pasta_base, config.wake_word)
    modelo = carregar(arquivo, pasta_base)
    log.info("Wake word: \"%s\" (modelo %s, limiar %.2f, cooldown %.1f s)",
             frase, arquivo.name, config.wake_threshold, config.wake_cooldown_s)
    return DetectorPalavra(modelo, nome, frase, config.wake_threshold, config.wake_cooldown_s)


# ---------------------------------------------------------------- bip

def gerar_bip(frequencia: float = 880.0, duracao: float = 0.12, volume: float = 0.3,
              taxa: int = TAXA) -> np.ndarray:
    """Bip senoidal curto com rampa de 5 ms nas pontas (sem estalo)."""
    n = int(duracao * taxa)
    t = np.arange(n) / taxa
    onda = volume * np.sin(2 * np.pi * frequencia * t)
    rampa = min(int(0.005 * taxa), n // 2)
    if rampa:
        envelope = np.ones(n)
        envelope[:rampa] = np.linspace(0, 1, rampa)
        envelope[-rampa:] = np.linspace(1, 0, rampa)
        onda *= envelope
    return onda.astype(np.float32)


def tocar_bip() -> None:
    import sounddevice as sd

    try:
        sd.play(gerar_bip(), TAXA)  # não bloqueia
    except sd.PortAudioError as erro:
        log.warning("Não consegui tocar o bip (%s); a detecção continua.", erro)


# ---------------------------------------------------------------- comando

def comando_escutar(config, dispositivo: int | None, verificador=None, gravador=None,
                    salvar_comando: Path | None = None) -> int:
    """Detecta a palavra-chave e loga o score.

    Com `gravador` (Bloco 3), grava o comando depois do bip até o silêncio; com `verificador` ativo,
    confere a voz usando palavra-chave + comando (o mesmo trecho que o loop principal vai usar).
    `salvar_comando` grava o último comando num WAV para conferir o corte (só quando pedido).
    """
    detector = criar_detector(config)
    if verificador is not None:
        verificador.preparar()
    print(f"Escutando... diga \"{detector.frase}\" (Ctrl+C para sair)", flush=True)
    pico = 0.0
    inicio_janela = time.monotonic()
    try:
        with Microfone(dispositivo) as mic:
            for frame in mic.frames():
                deteccao = detector.processar(frame)
                if deteccao is not None:
                    log.info("Palavra-chave detectada! score=%.3f", deteccao.score)
                    tocar_bip()
                    if gravador is not None:
                        _gravar_e_verificar(mic, detector, gravador, verificador, salvar_comando)
                    pico = 0.0
                    inicio_janela = time.monotonic()
                    continue

                pico = max(pico, detector.ultimo_score)
                if time.monotonic() - inicio_janela >= 1.0:
                    # Quase-detecções em INFO ajudam a calibrar o WAKE_THRESHOLD.
                    if pico >= detector.limiar / 2:
                        log.info("score máximo no último segundo: %.3f (limiar %.2f)", pico, detector.limiar)
                    else:
                        log.debug("score máximo no último segundo: %.3f", pico)
                    pico = 0.0
                    inicio_janela = time.monotonic()
    except KeyboardInterrupt:
        print()
    return 0


def _gravar_e_verificar(mic, detector: DetectorPalavra, gravador, verificador, salvar_comando: Path | None) -> None:
    def frames_alimentando_detector():
        # O detector continua vendo o áudio: o fluxo do modelo não fica com buraco depois do comando.
        for frame in mic.frames():
            detector.processar(frame)
            yield frame

    resultado = gravador.gravar(frames_alimentando_detector(), detector.audio_recente())
    if not resultado.teve_fala:
        return
    if salvar_comando is not None:
        from chubasa.audio import salvar_wav

        salvar_wav(salvar_comando, resultado.audio_comando)
        log.info("comando salvo em %s", salvar_comando)
    if verificador is not None and verificador.ativo:
        verificador.verificar(resultado.audio_palavra, resultado.audio_comando)
