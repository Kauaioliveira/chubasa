"""Ícone na bandeja: o assistente roda em segundo plano, sem janela.

`pythonw -m chubasa --bandeja` (ou `chubasa-bandeja`): o ícone muda de cor conforme o estado e o
menu permite pausar/retomar o microfone e sair. O log vai para arquivo (veja config.pasta_de_logs).
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from typing import Any

from PIL import Image, ImageDraw

from chubasa.loop import Estado

log = logging.getLogger(__name__)

TAMANHO = 64
CORES = {
    Estado.PAUSADO: (150, 150, 150),
    Estado.DORMINDO: (52, 120, 246),
    Estado.OUVINDO: (40, 180, 99),
    Estado.CONVERSANDO: (40, 180, 99),
    Estado.PENSANDO: (243, 156, 18),
    Estado.FALANDO: (155, 89, 182),
}
TITULOS = {
    Estado.PAUSADO: "Chubasa: microfone pausado",
    Estado.DORMINDO: "Chubasa: esperando a palavra-chave",
    Estado.OUVINDO: "Chubasa: ouvindo",
    Estado.CONVERSANDO: "Chubasa: ouvindo (conversa)",
    Estado.PENSANDO: "Chubasa: pensando",
    Estado.FALANDO: "Chubasa: falando",
}


def desenhar_icone(estado: Estado) -> Image.Image:
    """Círculo na cor do estado com um microfone branco (riscado quando pausado)."""
    imagem = Image.new("RGBA", (TAMANHO, TAMANHO), (0, 0, 0, 0))
    d = ImageDraw.Draw(imagem)
    d.ellipse((2, 2, TAMANHO - 3, TAMANHO - 3), fill=CORES[estado])
    branco = (255, 255, 255, 255)
    d.rounded_rectangle((25, 13, 39, 37), radius=7, fill=branco)  # cápsula
    d.arc((19, 23, 45, 45), start=0, end=180, fill=branco, width=3)  # suporte
    d.line((32, 45, 32, 51), fill=branco, width=3)
    d.line((25, 51, 39, 51), fill=branco, width=3)
    if estado == Estado.PAUSADO:
        d.line((16, 16, 48, 48), fill=(200, 30, 30, 255), width=5)
    return imagem


class Bandeja:
    """Liga o ícone (thread principal) ao assistente (thread própria com loop asyncio)."""

    def __init__(self, criar_assistente: Callable[[Callable[[Estado], None]], Any],
                 criar_icone: Callable[..., Any] | None = None, criar_menu: Callable[[], Any] | None = None):
        self._criar_assistente = criar_assistente
        self._criar_icone = criar_icone
        self._criar_menu = criar_menu if criar_menu is not None else self._menu
        self.assistente: Any = None
        self.icone: Any = None
        self._thread: threading.Thread | None = None
        self._saindo = threading.Event()
        self.erro: BaseException | None = None

    # -- menu

    def _alternar_pausa(self, _icone=None, _item=None) -> None:
        if self.assistente is None:
            return
        if self.assistente.pausado:
            self.assistente.retomar()
            log.info("microfone retomado pela bandeja")
        else:
            self.assistente.pausar()
            log.info("microfone pausado pela bandeja")
        self.ao_mudar_estado(self.assistente.estado if not self.assistente.pausado else Estado.PAUSADO)

    def sair(self, _icone=None, _item=None) -> None:
        log.info("saindo pela bandeja")
        self._saindo.set()
        if self.assistente is not None:
            self.assistente.parar()
        if self.icone is not None:
            self.icone.stop()

    def ao_mudar_estado(self, estado: Estado) -> None:
        if self.icone is None:
            return
        self.icone.icon = desenhar_icone(estado)
        self.icone.title = TITULOS[estado]

    # -- execução

    def _menu(self) -> Any:
        import pystray

        return pystray.Menu(
            pystray.MenuItem(lambda _i: "Retomar microfone" if self.assistente and self.assistente.pausado
                             else "Pausar microfone", self._alternar_pausa, default=True),
            pystray.MenuItem("Sair", self.sair),
        )

    def _rodar_assistente(self) -> None:
        try:
            # Os modelos carregam aqui (15 a 25 s), com o ícone já visível em "carregando".
            self.assistente = self._criar_assistente(self.ao_mudar_estado)
            if self._saindo.is_set():
                return
            self.ao_mudar_estado(Estado.DORMINDO)
            asyncio.run(self.assistente.rodar())
        except BaseException as erro:  # registra qualquer queda; a bandeja fecha junto
            self.erro = erro
            log.exception("o assistente parou com erro")
        finally:
            if self.icone is not None:
                self.icone.stop()

    def rodar(self, ao_ficar_visivel: Callable[[Any], None] | None = None) -> None:
        """Bloqueia até 'Sair'. `ao_ficar_visivel` serve para testes e automação."""
        if self._criar_icone is None:
            import pystray

            self._criar_icone = pystray.Icon
        self.icone = self._criar_icone("chubasa", desenhar_icone(Estado.PAUSADO), "Chubasa: carregando...",
                                       self._criar_menu())
        self._thread = threading.Thread(target=self._rodar_assistente, name="assistente", daemon=True)

        def preparar(icone: Any) -> None:
            icone.visible = True
            self._thread.start()
            if ao_ficar_visivel is not None:
                ao_ficar_visivel(icone)

        self.icone.run(setup=preparar)
        if self.assistente is not None:
            self.assistente.parar()
        if self._thread.is_alive():
            self._thread.join(timeout=10)


def comando_bandeja(config, dispositivo: int | None) -> int:
    from chubasa.loop import criar_assistente

    falante = []

    def criar(ao_mudar_estado):
        assistente = criar_assistente(config, dispositivo, ao_mudar_estado)
        falante.append(assistente.c.falante)
        return assistente

    bandeja = Bandeja(criar)
    try:
        bandeja.rodar()
    finally:
        for f in falante:
            f.fechar()
    return 1 if bandeja.erro is not None else 0
