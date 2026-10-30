"""Loop principal: dormindo -> ouvindo -> pensando -> falando -> (conversando) -> dormindo.

- Dormindo: só o detector da palavra-chave ouve. Enquanto o assistente fala (resposta ou aviso
  de timer), a palavra-chave fica muda para ele não se ativar com a própria voz.
- Ouvindo: grava o comando com o VAD, confere a voz (palavra-chave + comando) e transcreve.
- Pensando/falando: a resposta chega em streaming e é falada frase por frase.
- Interrupção durante a fala: só com voz cadastrada, score alto da palavra-chave e a voz
  verificada (a própria voz do assistente não passa na verificação).
- Conversando: depois de responder, escuta por alguns segundos sem precisar da palavra-chave.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import threading
import time
import unicodedata
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from chubasa.audio import DURACAO_FRAME, ErroDeAudio

log = logging.getLogger(__name__)

RABO_DE_ECO_S = 0.3  # áudio descartado depois que o assistente para de falar
FALA_ESQUECI = "Pronto, esqueci a nossa conversa."


class Estado(str, Enum):
    PAUSADO = "pausado"
    DORMINDO = "dormindo"
    OUVINDO = "ouvindo"
    PENSANDO = "pensando"
    FALANDO = "falando"
    CONVERSANDO = "conversando"


# ---------------------------------------------------------------- comandos de voz

_COMANDOS = {
    "esquecer": {"esquece", "esqueca", "esquece tudo", "esqueca tudo", "esquece a conversa",
                 "esqueca a conversa", "nova conversa", "comeca de novo", "limpa a conversa"},
    "parar": {"para", "pare", "parar", "cancela", "cancelar", "chega", "silencio", "deixa pra la",
              "deixa para la", "nada", "nao precisa", "esquece isso", "fica quieto"},
}
_PALAVRA_CHAVE = re.compile(r"^(hey|hei|ei|rei|oi)\s+(chubasa|chubaza|chubaça|jarvis)\s*")


def normalizar(texto: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    limpo = re.sub(r"[^a-z0-9 ]+", " ", sem_acento.lower())
    limpo = re.sub(r"\s+", " ", limpo).strip()
    return _PALAVRA_CHAVE.sub("", limpo).strip()


_PREFIXO_PALAVRA_CHAVE = re.compile(r"^\s*(hey|hei|ei|rei|oi)[\s,!.]+(chubasa|chubaza|chubaça|jarvis)\b[\s,!.?]*",
                                    re.IGNORECASE)


def remover_palavra_chave(texto: str) -> str:
    """No modo conversa a pessoa pode repetir a palavra-chave; ela não faz parte da pergunta."""
    return _PREFIXO_PALAVRA_CHAVE.sub("", texto).strip()


def comando_de_voz(texto: str) -> str | None:
    """'esquecer', 'parar' ou None. Só vale se a frase for o comando (curta), não 'para onde vou?'."""
    frase = normalizar(texto)
    for nome, variantes in _COMANDOS.items():
        if frase in variantes:
            return nome
    return None


# ---------------------------------------------------------------- assistente

@dataclass
class Componentes:
    detector: Any  # wakeword.DetectorPalavra
    gravador: Any  # vad.GravadorComando
    verificador: Any  # speaker.VerificadorVoz
    transcritor: Any  # stt.Transcritor
    cerebro: Any  # brain.Cerebro
    falante: Any  # tts.Falante
    abrir_microfone: Callable[[], Any]  # devolve um context manager com ler()/descartar_pendentes()
    bip: Callable[[], None]


class Assistente:
    def __init__(self, componentes: Componentes, conversa_s: float = 6.0, interrupcao: bool = True,
                 limiar_interrupcao: float = 0.8, ao_mudar_estado: Callable[[Estado], None] | None = None):
        self.c = componentes
        self.conversa_s = conversa_s
        self.limiar_interrupcao = limiar_interrupcao
        self.interrupcao = interrupcao and componentes.verificador.ativo
        if interrupcao and not componentes.verificador.ativo:
            log.info("Interromper a fala com a palavra-chave fica desligado até cadastrar a voz "
                     "(sem cadastro, o assistente poderia se interromper com a própria voz).")
        self._ao_mudar_estado = ao_mudar_estado
        self.estado = Estado.DORMINDO
        self._parar = threading.Event()
        self._pausado = threading.Event()
        self._mic: Any = None
        self._tarefa_resposta: asyncio.Task | None = None
        self._calar_resposta = False
        self.interacoes = 0

    # -- controle (seguro de chamar de outra thread, ex.: bandeja)

    def parar(self) -> None:
        self._parar.set()

    def pausar(self) -> None:
        self._pausado.set()

    def retomar(self) -> None:
        self._pausado.clear()

    @property
    def pausado(self) -> bool:
        return self._pausado.is_set()

    def _mudar(self, estado: Estado) -> None:
        if estado != self.estado:
            log.debug("estado: %s -> %s", self.estado.value, estado.value)
            self.estado = estado
            if self._ao_mudar_estado is not None:
                try:
                    self._ao_mudar_estado(estado)
                except Exception as erro:  # um observador quebrado não pode parar o assistente
                    log.warning("erro ao avisar mudança de estado: %s", erro)

    # -- laço principal

    async def rodar(self) -> None:
        try:
            await self._abrir_cerebro()
            while not self._parar.is_set():
                if self._pausado.is_set():
                    self._mudar(Estado.PAUSADO)
                    await asyncio.sleep(0.2)
                    continue
                try:
                    with self.c.abrir_microfone() as mic:
                        self._mic = mic
                        self._mudar(Estado.DORMINDO)
                        log.info("Pronto: diga \"%s\"", self.c.detector.frase)
                        await self._dormir()
                except ErroDeAudio as erro:
                    if self._parar.is_set():
                        break
                    log.error("Problema no microfone (%s); tentando de novo em 3 s.", erro)
                    await asyncio.sleep(3)
                finally:
                    self._mic = None
        finally:
            await self._encerrar()

    async def _abrir_cerebro(self) -> None:
        try:
            await self.c.cerebro.iniciar()
        except Exception as erro:  # sem login/rede: tenta de novo na primeira pergunta
            log.warning("Não consegui abrir a conversa agora (%s); tento de novo na primeira pergunta.", erro)

    async def _encerrar(self) -> None:
        if self.c.falante.falando.is_set():
            self.c.falante.interromper()
        if self._tarefa_resposta is not None and not self._tarefa_resposta.done():
            self._tarefa_resposta.cancel()
        await self.c.cerebro.fechar()
        log.info("Assistente encerrado.")

    async def _ler(self, timeout: float = 1.0) -> np.ndarray | None:
        try:
            return await asyncio.to_thread(self._mic.ler, timeout)
        except ErroDeAudio:
            if self._parar.is_set():
                return None
            raise

    async def _dormir(self) -> None:
        """Volta daqui só para pausar, parar ou reabrir o microfone."""
        estava_falando = False
        while not self._parar.is_set() and not self._pausado.is_set():
            frame = await self._ler()
            if frame is None:
                continue
            if self.c.falante.falando.is_set():  # aviso de timer etc.: não ouvir a própria voz
                estava_falando = True
                continue
            if estava_falando:
                await self._descartar_eco()
                estava_falando = False
                continue
            deteccao = self.c.detector.processar(frame)
            if deteccao is not None:
                log.info("Palavra-chave detectada (score %.2f)", deteccao.score)
                await self._atender(self.c.detector.audio_recente())
                self._mudar(Estado.DORMINDO)

    async def _atender(self, audio_palavra: np.ndarray) -> None:
        self.c.bip()
        palavra: np.ndarray | None = audio_palavra
        while not self._parar.is_set():
            conversa = palavra is None
            resultado = await self._interagir(
                palavra if palavra is not None else np.zeros(0, dtype=np.int16),
                timeout=self.conversa_s if conversa else None,
                conversa=conversa,
            )
            await self._descartar_eco()
            if resultado == "interrompido":
                self.c.bip()
                palavra = self.c.detector.audio_recente()
                continue
            if resultado == "respondeu" and self.conversa_s > 0:
                palavra = None  # modo conversa: a próxima fala não precisa da palavra-chave
                continue
            return

    async def _interagir(self, palavra: np.ndarray, timeout: float | None, conversa: bool) -> str:
        """Grava, confere a voz, transcreve e responde. Devolve 'nada', 'respondeu' ou 'interrompido'."""
        self._mudar(Estado.CONVERSANDO if conversa else Estado.OUVINDO)
        gravacao = await asyncio.to_thread(self.c.gravador.gravar, self._frames_da_thread(), palavra, timeout)
        if not gravacao.teve_fala or self._parar.is_set():
            return "nada"

        voz = await asyncio.to_thread(self.c.verificador.verificar, gravacao.audio_palavra, gravacao.audio_comando)
        if not voz.aceito:
            log.info("Voz não reconhecida: ignorando o comando.")
            return "nada"

        self._mudar(Estado.PENSANDO)
        transcricao = await asyncio.to_thread(self.c.transcritor.transcrever, gravacao.audio_comando)
        texto = remover_palavra_chave(transcricao.texto)
        if not texto:
            return "nada"
        log.info("Você: %s", texto)

        comando = comando_de_voz(texto)
        if comando == "parar":
            self.c.falante.interromper()
            return "nada"
        if comando == "esquecer":
            await self._esperar_resposta_anterior()
            try:
                await self.c.cerebro.esquecer()
            except Exception as erro:
                log.warning("Não consegui reabrir a conversa (%s); ela reabre na próxima pergunta.", erro)
            await self._falar_e_esperar(FALA_ESQUECI)
            return "nada"

        self.interacoes += 1
        return await self._responder(texto)

    def _frames_da_thread(self) -> Iterator[np.ndarray]:
        """Frames para o gravador (roda numa thread). O detector continua vendo o áudio."""
        while not self._parar.is_set():
            frame = self._mic.ler(1.0)
            self.c.detector.processar(frame)
            yield frame

    async def _responder(self, texto: str) -> str:
        from chubasa.brain import frases

        await self._esperar_resposta_anterior()
        self._calar_resposta = False

        async def produzir() -> None:
            async for frase in frases(self.c.cerebro.responder(texto)):
                if self._calar_resposta:
                    continue  # interrompido: só termina de ler a resposta para o histórico ficar certo
                self._mudar(Estado.FALANDO)
                log.info("Assistente: %s", frase)
                self.c.falante.falar(frase)

        self._tarefa_resposta = asyncio.create_task(produzir())
        while not self._tarefa_resposta.done() or self.c.falante.falando.is_set():
            if self._parar.is_set():
                self._calar()
                break
            frame = await self._ler(timeout=0.2)
            if frame is None or not self.interrupcao:
                continue
            deteccao = self.c.detector.processar(frame)
            if deteccao is None or deteccao.score < self.limiar_interrupcao:
                continue
            voz = await asyncio.to_thread(self.c.verificador.verificar, self.c.detector.audio_recente())
            if voz.aceito:
                log.info("Interrompido pela palavra-chave (score %.2f).", deteccao.score)
                self._calar()
                return "interrompido"
        if self._tarefa_resposta.done() and self._tarefa_resposta.exception() is not None:
            log.error("Erro ao responder: %s", self._tarefa_resposta.exception())
        return "respondeu"

    def _calar(self) -> None:
        self._calar_resposta = True
        self.c.falante.interromper()

    async def _esperar_resposta_anterior(self) -> None:
        tarefa = self._tarefa_resposta
        if tarefa is not None and not tarefa.done():
            try:
                await asyncio.wait_for(asyncio.shield(tarefa), timeout=self.c.cerebro.timeout_total_s)
            except (asyncio.TimeoutError, Exception) as erro:
                log.warning("A resposta anterior não terminou (%s); seguindo.", erro)

    async def _falar_e_esperar(self, texto: str) -> None:
        self._mudar(Estado.FALANDO)
        self.c.falante.falar(texto)
        while self.c.falante.falando.is_set() and not self._parar.is_set():
            await self._ler(timeout=0.2)  # continua drenando o microfone

    async def _descartar_eco(self) -> None:
        """Depois de falar: joga fora o áudio da própria voz e o rabo de eco da sala."""
        if self._mic is None:
            return
        self._mic.descartar_pendentes()
        for _ in range(math.ceil(RABO_DE_ECO_S / DURACAO_FRAME)):  # conta frames: vale em tempo de áudio
            if self._parar.is_set() or await self._ler(timeout=DURACAO_FRAME * 3) is None:
                break
        self.c.detector.reiniciar()


# ---------------------------------------------------------------- montagem

def criar_assistente(config, dispositivo: int | None, ao_mudar_estado=None) -> Assistente:
    """Carrega todos os modelos uma vez, antes de começar a ouvir."""
    from chubasa.audio import Microfone
    from chubasa.brain import criar_cerebro
    from chubasa.speaker import criar_verificador
    from chubasa.stt import criar_transcritor
    from chubasa.tts import criar_falante
    from chubasa.vad import criar_gravador
    from chubasa.wakeword import criar_detector, tocar_bip

    inicio = time.perf_counter()
    detector = criar_detector(config)
    gravador = criar_gravador(config)
    verificador = criar_verificador(config)
    verificador.preparar()
    transcritor = criar_transcritor(config)
    falante = criar_falante(config)
    cerebro = criar_cerebro(config)
    log.info("Modelos carregados em %.1f s", time.perf_counter() - inicio)
    componentes = Componentes(detector, gravador, verificador, transcritor, cerebro, falante,
                              lambda: Microfone(dispositivo), tocar_bip)
    return Assistente(componentes, config.loop_conversation_s, config.loop_barge_in,
                      config.loop_barge_in_threshold, ao_mudar_estado)


def comando_assistente(config, dispositivo: int | None) -> int:
    assistente = criar_assistente(config, dispositivo)
    print("Chubasa rodando. Ctrl+C para sair.", flush=True)
    try:
        asyncio.run(assistente.rodar())
    except KeyboardInterrupt:
        pass
    finally:
        assistente.c.falante.fechar()
    print("Até logo!")
    return 0
