from __future__ import annotations

import pytest

from chubasa import config

VARIAVEIS = (
    "WAKE_WORD",
    "LANGUAGE",
    "MIC_DEVICE",
    "LOG_LEVEL",
    "MODELS_DIR",
    "WAKE_MODEL_PATH",
    "WAKE_THRESHOLD",
    "WAKE_COOLDOWN_S",
    "SPEAKER_VERIFY",
    "SPEAKER_PROFILE_PATH",
    "SPEAKER_THRESHOLD",
    "VAD_THRESHOLD",
    "VAD_SILENCE_END_S",
    "VAD_NO_SPEECH_TIMEOUT_S",
    "VAD_MAX_RECORDING_S",
    "VAD_PREROLL_S",
    "STT_MODEL",
    "STT_DEVICE",
    "STT_COMPUTE_TYPE",
    "STT_BEAM_SIZE",
    "STT_CPU_THREADS",
    "BRAIN_MODEL",
    "BRAIN_TIMEOUT_FIRST_S",
    "BRAIN_TIMEOUT_TOTAL_S",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_API_KEY",
)


@pytest.fixture(autouse=True)
def ambiente_isolado(monkeypatch, tmp_path):
    """Os testes nunca leem o .env real nem herdam variáveis da máquina."""
    for nome in VARIAVEIS:
        monkeypatch.delenv(nome, raising=False)
    monkeypatch.setattr(config, "ARQUIVO_ENV", tmp_path / "nao-existe.env")


@pytest.fixture
def env_minimo(monkeypatch):
    monkeypatch.setenv("WAKE_WORD", "hey chubasa")
    monkeypatch.setenv("LANGUAGE", "pt")
