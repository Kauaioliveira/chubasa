from __future__ import annotations

import logging
from pathlib import Path

import pytest

from chubasa import config
from chubasa.__main__ import main


def test_carrega_do_arquivo_env(tmp_path):
    arquivo = tmp_path / ".env"
    arquivo.write_text("WAKE_WORD=hey chubasa\nLANGUAGE=pt\nMIC_DEVICE=\nLOG_LEVEL=debug\n", encoding="utf-8")

    cfg = config.carregar_config(arquivo)

    assert cfg.wake_word == "hey chubasa"
    assert cfg.language == "pt"
    assert cfg.mic_device is None
    assert cfg.log_level == "DEBUG"


def test_variavel_de_ambiente_vence_o_arquivo(tmp_path, monkeypatch):
    arquivo = tmp_path / ".env"
    arquivo.write_text("WAKE_WORD=hey chubasa\nLANGUAGE=pt\n", encoding="utf-8")
    monkeypatch.setenv("LANGUAGE", "en")

    assert config.carregar_config(arquivo).language == "en"


def test_falta_obrigatorio_da_mensagem_clara(tmp_path):
    with pytest.raises(config.ConfiguracaoInvalida) as erro:
        config.carregar_config(tmp_path / "nao-existe.env")

    mensagem = str(erro.value)
    assert "WAKE_WORD: obrigatório" in mensagem
    assert "LANGUAGE: obrigatório" in mensagem
    assert ".env.example" in mensagem


def test_nivel_de_log_invalido(env_minimo, monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "barulhento")

    with pytest.raises(config.ConfiguracaoInvalida, match="LOG_LEVEL"):
        config.carregar_config()


def test_caminho_relativo_e_resolvido_na_raiz():
    assert config.caminho_do_projeto(Path("models/x.onnx")) == config.RAIZ / "models" / "x.onnx"
    absoluto = config.RAIZ / "outro"
    assert config.caminho_do_projeto(absoluto) == absoluto


def test_avisa_alto_se_chave_api_existir(caplog):
    with caplog.at_level(logging.WARNING):
        assert config.avisar_chave_api({"ANTHROPIC_API_KEY": "sk-teste"}) is True
    assert "ANTHROPIC_API_KEY" in caplog.text
    assert "sk-teste" not in caplog.text


def test_sem_chave_api_nao_avisa(caplog):
    assert config.avisar_chave_api({}) is False
    assert caplog.text == ""


def test_main_sem_comando_imprime_ok(env_minimo, capsys):
    assert main([]) == 0
    assert capsys.readouterr().out.strip() == "configuração ok"


def test_main_sem_env_sai_com_erro(capsys):
    assert main([]) == 2
    assert "WAKE_WORD" in capsys.readouterr().err
