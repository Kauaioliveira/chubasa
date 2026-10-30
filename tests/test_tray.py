from __future__ import annotations

import asyncio
import logging
import threading
import time
from pathlib import Path

import pytest

from chubasa import config, tray
from chubasa.loop import Estado


class IconeFalso:
    def __init__(self, nome, icone, titulo, menu):
        self.nome, self.icon, self.title, self.menu = nome, icone, titulo, menu
        self.visible = False
        self._parado = threading.Event()

    def run(self, setup):
        setup(self)
        self._parado.wait(10)

    def stop(self):
        self._parado.set()


class AssistenteFalso:
    def __init__(self, ao_mudar_estado, falhar=False):
        self.ao_mudar_estado = ao_mudar_estado
        self.falhar = falhar
        self.estado = Estado.DORMINDO
        self._parar = threading.Event()
        self._pausado = threading.Event()
        self.rodou = threading.Event()

    @property
    def pausado(self):
        return self._pausado.is_set()

    def pausar(self):
        self._pausado.set()

    def retomar(self):
        self._pausado.clear()

    def parar(self):
        self._parar.set()

    async def rodar(self):
        self.rodou.set()
        if self.falhar:
            raise RuntimeError("caiu")
        while not self._parar.is_set():
            await asyncio.sleep(0.02)


def test_icone_tem_cor_por_estado():
    cores = set()
    for estado in Estado:
        imagem = tray.desenhar_icone(estado)
        assert imagem.size == (64, 64) and imagem.mode == "RGBA"
        cores.add(imagem.getpixel((8, 32)))
        assert estado in tray.TITULOS
    assert len(cores) >= 5  # ouvindo e conversando dividem a cor


def test_icone_pausado_e_riscado():
    assert tray.desenhar_icone(Estado.PAUSADO).getpixel((18, 18))[:3] == (200, 30, 30)


def test_bandeja_liga_estado_pausa_e_sair():
    criados = []

    def criar(ao_mudar):
        criados.append(AssistenteFalso(ao_mudar))
        return criados[0]

    bandeja = tray.Bandeja(criar, criar_icone=IconeFalso, criar_menu=lambda: None)
    observado = {}

    def visivel(icone):
        assert criados[0].rodou.wait(2)
        criados[0].ao_mudar_estado(Estado.FALANDO)
        observado["titulo_falando"] = icone.title
        bandeja._alternar_pausa()
        observado["pausado"] = criados[0].pausado
        observado["titulo_pausado"] = icone.title
        bandeja._alternar_pausa()
        observado["retomado"] = not criados[0].pausado
        bandeja.sair()

    inicio = time.perf_counter()
    bandeja.rodar(ao_ficar_visivel=visivel)

    assert time.perf_counter() - inicio < 5
    assert bandeja.icone.visible
    assert observado == {"titulo_falando": "Chubasa: falando", "pausado": True,
                         "titulo_pausado": "Chubasa: microfone pausado", "retomado": True}
    assert criados[0]._parar.is_set()
    assert not bandeja._thread.is_alive()
    assert bandeja.erro is None


def test_assistente_que_cai_fecha_a_bandeja(caplog):
    bandeja = tray.Bandeja(lambda ao: AssistenteFalso(ao, falhar=True), criar_icone=IconeFalso, criar_menu=lambda: None)
    bandeja.rodar()
    assert isinstance(bandeja.erro, RuntimeError)
    assert "o assistente parou com erro" in caplog.text


def test_pasta_de_logs_por_sistema(tmp_path):
    assert config.pasta_de_logs({"LOCALAPPDATA": str(tmp_path)}, "win32") == tmp_path / "Chubasa" / "logs"
    assert config.pasta_de_logs({"XDG_STATE_HOME": str(tmp_path)}, "linux") == tmp_path / "chubasa"
    assert config.pasta_de_logs({}, "linux") == Path.home() / ".local" / "state" / "chubasa"


def test_log_em_arquivo_com_rotacao(tmp_path):
    arquivo = tmp_path / "logs" / "chubasa.log"
    try:
        config.configurar_logging("INFO", arquivo, console=False)
        logging.getLogger("teste").info("mensagem com acentuação")
        rotativos = [h for h in logging.getLogger().handlers if hasattr(h, "maxBytes")]
        assert rotativos and rotativos[0].maxBytes == 1_000_000 and rotativos[0].backupCount == 5
        assert not any(type(h) is logging.StreamHandler for h in logging.getLogger().handlers)
        for h in logging.getLogger().handlers:
            h.flush()
        assert "mensagem com acentuação" in arquivo.read_text(encoding="utf-8")
    finally:
        config.configurar_logging("INFO")


@pytest.mark.interface
def test_bandeja_real_abre_e_fecha():
    pytest.importorskip("pystray")
    criados = []

    def criar(ao):
        criados.append(AssistenteFalso(ao))
        return criados[0]

    bandeja = tray.Bandeja(criar)

    def visivel(_icone):
        threading.Timer(1.5, bandeja.sair).start()

    inicio = time.perf_counter()
    bandeja.rodar(ao_ficar_visivel=visivel)
    duracao = time.perf_counter() - inicio

    assert 1.0 < duracao < 10
    assert criados[0].rodou.is_set() and criados[0]._parar.is_set()
    assert bandeja.erro is None


def test_sair_durante_a_carga_nao_roda_o_assistente():
    liberar = threading.Event()
    criados = []

    def criar_devagar(ao):
        liberar.wait(5)
        criados.append(AssistenteFalso(ao))
        return criados[0]

    bandeja = tray.Bandeja(criar_devagar, criar_icone=IconeFalso, criar_menu=lambda: None)

    def visivel(icone):
        assert icone.title == "Chubasa: carregando..."
        bandeja.sair()
        liberar.set()

    bandeja.rodar(ao_ficar_visivel=visivel)
    bandeja._thread.join(5)
    assert criados and not criados[0].rodou.is_set()
