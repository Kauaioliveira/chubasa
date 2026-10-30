"""Configuração do Chubasa: lê o `.env`, valida e prepara o logging."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

RAIZ = Path(__file__).resolve().parent.parent
ARQUIVO_ENV = RAIZ / ".env"

NIVEIS_LOG = ("DEBUG", "INFO", "WARNING", "ERROR")

log = logging.getLogger(__name__)


class ConfiguracaoInvalida(Exception):
    """O `.env` está faltando ou tem valor inválido."""


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ARQUIVO_ENV,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    wake_word: str = Field(min_length=1)
    language: str = Field(min_length=2)
    mic_device: str | None = None
    log_level: str = "INFO"

    # Pasta de modelos baixados/treinados (fora do git)
    models_dir: Path = Path("models")

    # Wake word (Bloco 2)
    wake_model_path: Path = Path("models/hey_chubasa.onnx")
    wake_threshold: float = Field(default=0.5, gt=0, lt=1)
    wake_cooldown_s: float = Field(default=2.0, ge=0)

    # Verificação de voz (Bloco 2.5)
    speaker_verify: bool = True
    speaker_profile_path: Path = Path("models/voz.npy")
    speaker_threshold: float = Field(default=0.40, gt=-1, lt=1)

    # Gravação do comando com VAD (Bloco 3)
    vad_threshold: float = Field(default=0.5, gt=0, lt=1)
    vad_silence_end_s: float = Field(default=1.0, gt=0)
    vad_no_speech_timeout_s: float = Field(default=5.0, gt=0)
    vad_max_recording_s: float = Field(default=30.0, gt=0)
    vad_preroll_s: float = Field(default=0.3, ge=0)

    # Fala para texto (Bloco 4)
    stt_model: str = "small"
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"
    stt_beam_size: int = Field(default=1, ge=1, le=10)
    stt_cpu_threads: int = Field(default=0, ge=0)

    @field_validator("stt_device")
    @classmethod
    def _dispositivo_stt(cls, valor: str) -> str:
        valor = valor.strip().lower()
        if valor not in ("cpu", "cuda", "auto"):
            raise ValueError("use cpu, cuda ou auto")
        return valor

    @field_validator("models_dir", "wake_model_path", "speaker_profile_path")
    @classmethod
    def _relativo_a_raiz(cls, valor: Path) -> Path:
        return caminho_do_projeto(valor)

    @field_validator("mic_device", mode="before")
    @classmethod
    def _vazio_vira_none(cls, valor: object) -> object:
        if isinstance(valor, str) and not valor.strip():
            return None
        return valor

    @field_validator("log_level")
    @classmethod
    def _nivel_valido(cls, valor: str) -> str:
        nivel = valor.strip().upper()
        if nivel not in NIVEIS_LOG:
            raise ValueError(f"use um destes: {', '.join(NIVEIS_LOG)}")
        return nivel


def caminho_do_projeto(caminho: Path) -> Path:
    """Caminho relativo no `.env` é relativo à raiz do projeto, não ao diretório atual."""
    return caminho if caminho.is_absolute() else RAIZ / caminho


def carregar_config(env_file: Path | None = None) -> Config:
    arquivo = env_file if env_file is not None else ARQUIVO_ENV
    try:
        return Config(_env_file=arquivo)
    except ValidationError as erro:
        raise ConfiguracaoInvalida(_mensagem_de_erro(erro, arquivo)) from None


def _mensagem_de_erro(erro: ValidationError, arquivo: Path) -> str:
    linhas = [f"Configuração inválida (arquivo: {arquivo}):"]
    for item in erro.errors():
        campo = str(item["loc"][0]).upper() if item["loc"] else "?"
        motivo = "obrigatório e não foi definido" if item["type"] == "missing" else item["msg"]
        linhas.append(f"  - {campo}: {motivo}")
    if not arquivo.exists():
        linhas.append(
            "O arquivo .env não existe. Crie a partir do exemplo: "
            "Copy-Item .env.example .env (PowerShell) ou cp .env.example .env"
        )
    return "\n".join(linhas)


# Bibliotecas que falam demais em INFO (requisições HTTP, progresso); só aparecem em DEBUG.
BIBLIOTECAS_RUIDOSAS = ("httpx", "httpx2", "huggingface_hub", "faster_whisper", "speechbrain")


def configurar_logging(nivel: str = "INFO") -> None:
    logging.basicConfig(
        level=nivel,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    for nome in BIBLIOTECAS_RUIDOSAS:
        logging.getLogger(nome).setLevel(logging.DEBUG if nivel == "DEBUG" else logging.WARNING)


def avisar_chave_api(ambiente: Mapping[str, str] | None = None) -> bool:
    """Avisa se ANTHROPIC_API_KEY estiver definida: com ela o uso vira cobrança por token."""
    ambiente = os.environ if ambiente is None else ambiente
    if not ambiente.get("ANTHROPIC_API_KEY"):
        return False
    log.warning("=" * 70)
    log.warning("ATENÇÃO: ANTHROPIC_API_KEY está definida nas variáveis de ambiente.")
    log.warning("Com ela, o uso do Claude é cobrado por token em vez de usar o plano Pro.")
    log.warning("Remova a variável (Painel de Controle > Variáveis de ambiente) e reabra o terminal.")
    log.warning("=" * 70)
    return True


def iniciar(env_file: Path | None = None) -> Config:
    """Carrega a configuração, liga o logging e faz as checagens de segurança."""
    config = carregar_config(env_file)
    configurar_logging(config.log_level)
    avisar_chave_api()
    return config
