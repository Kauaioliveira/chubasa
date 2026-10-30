"""Conversa com o Claude pelo Claude Agent SDK, usando o login do plano Pro (sem chave de API).

Toda chamada ao modelo fica aqui: para trocar por outra forma de acesso, só este módulo muda.
A resposta chega em streaming (`responder`) e `frases` corta o texto em frases completas para o TTS.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import socket
import sys
import tempfile
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = """\
Você é um assistente de voz em português do Brasil. Tudo o que você escreve é lido em voz alta \
por um sintetizador de fala, então escreva como quem fala.
Regras:
- Responda curto e direto: de uma a três frases, a não ser que peçam detalhes.
- Nunca use markdown, listas, tópicos, títulos, tabelas, emojis, código ou símbolos especiais.
- Escreva números, horas e siglas de um jeito natural de falar.
- Nunca diga a palavra "Chubasa" nem qualquer nome para si mesmo. Se perguntarem seu nome, \
diga que é o assistente de voz da casa.
- Você não tem acesso a internet, arquivos, relógio nem ferramentas. Se precisar disso para \
responder, diga numa frase que ainda não consegue fazer isso.
- Use o que já foi dito na conversa quando a pergunta depender disso.
"""

# Ferramentas embutidas que nunca podem rodar (além de `tools=[]`, que já desliga todas)
FERRAMENTAS_BLOQUEADAS = [
    "Bash", "BashOutput", "KillShell", "Read", "Write", "Edit", "MultiEdit", "NotebookEdit",
    "Glob", "Grep", "LS", "WebFetch", "WebSearch", "Task", "Agent", "TodoWrite", "Skill",
    "SlashCommand", "ExitPlanMode",
]
# Variáveis que fariam o uso ser cobrado por token (chaves) ou desviariam as chamadas e o login
# do plano para outro servidor (BASE_URL). O processo do Claude nunca deve recebê-las.
VARIAVEIS_DE_CHAVE = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL")

FALAS_DE_ERRO = {
    "sem_internet": "Estou sem conexão com a internet agora. Tenta de novo daqui a pouco.",
    "limite": "Cheguei no limite de uso do plano por enquanto. Tenta de novo mais tarde.",
    "autenticacao": "Não estou conectado à conta. No computador, abra o terminal, rode claude e faça o login.",
    "timeout": "Demorei demais para pensar e desisti. Pode perguntar de novo?",
    "erro": "Tive um problema para responder agora. Pode tentar de novo?",
}

log = logging.getLogger(__name__)


class FalhaDoCerebro(Exception):
    def __init__(self, tipo: str, detalhe: str = ""):
        super().__init__(f"{tipo}: {detalhe}" if detalhe else tipo)
        self.tipo = tipo


# ---------------------------------------------------------------- ambiente

def remover_chaves_de_api() -> list[str]:
    """Tira as chaves do ambiente deste processo, para o processo do Claude não herdá-las."""
    removidas = [nome for nome in VARIAVEIS_DE_CHAVE if os.environ.pop(nome, None) is not None]
    if removidas:
        log.warning("Removi %s do ambiente do assistente: o Claude vai usar o login do plano.",
                    ", ".join(removidas))
    return removidas


def pasta_neutra() -> Path:
    """Diretório de trabalho do Claude fora do projeto (para não ler CLAUDE.md de ninguém)."""
    base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
    pasta = Path(base) / "Chubasa" / "conversa"
    pasta.mkdir(parents=True, exist_ok=True)
    return pasta


def tem_internet(host: str = "api.anthropic.com", porta: int = 443, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, porta), timeout=timeout):
            return True
    except OSError:
        return False


def criar_opcoes(modelo: str) -> Any:
    from claude_agent_sdk import ClaudeAgentOptions

    return ClaudeAgentOptions(
        model=modelo,
        system_prompt=SYSTEM_PROMPT,
        tools=[],  # nenhuma ferramenta embutida
        allowed_tools=[],
        disallowed_tools=list(FERRAMENTAS_BLOQUEADAS),
        setting_sources=[],  # não carrega settings nem CLAUDE.md de usuário/projeto
        mcp_servers={},
        strict_mcp_config=True,  # ignora servidores MCP configurados na máquina
        include_partial_messages=True,  # streaming de texto
        thinking={"type": "disabled"},  # menos latência numa conversa curta
        cwd=pasta_neutra(),
    )


def criar_cliente_sdk(opcoes: Any) -> Any:
    from claude_agent_sdk import ClaudeSDKClient

    return ClaudeSDKClient(options=opcoes)


# ---------------------------------------------------------------- cérebro

def _classificar(erro: BaseException) -> str:
    texto = str(erro).lower()
    if any(p in texto for p in ("rate limit", "usage limit", "limit reached", "rate_limit", "quota")):
        return "limite"
    if any(p in texto for p in ("login", "authenticat", "unauthorized", "401", "oauth")):
        return "autenticacao"
    if any(p in texto for p in ("connection", "network", "enotfound", "econnrefused", "getaddrinfo",
                                "timed out", "offline")):
        return "sem_internet"
    return "erro"


_ERRO_DO_SDK = {
    "rate_limit": "limite",
    "billing_error": "limite",
    "authentication_failed": "autenticacao",
}


class Cerebro:
    """Mantém um cliente do SDK aberto entre perguntas (histórico + menos latência).

    `responder` devolve um gerador assíncrono de pedaços de texto. Em caso de erro, em vez de
    exceção ele entrega uma frase falável (FALAS_DE_ERRO) e reconecta na próxima pergunta.
    """

    def __init__(self, modelo: str = "sonnet", timeout_primeiro_s: float = 30.0, timeout_total_s: float = 90.0,
                 criar_cliente: Callable[[Any], Any] = criar_cliente_sdk,
                 criar_opcoes_fn: Callable[[str], Any] = criar_opcoes,
                 checar_internet: Callable[[], bool] = tem_internet):
        self.modelo = modelo
        self.timeout_primeiro_s = timeout_primeiro_s
        self.timeout_total_s = timeout_total_s
        self._criar_cliente = criar_cliente
        self._criar_opcoes = criar_opcoes_fn
        self._checar_internet = checar_internet
        self._cliente: Any = None
        self.ultima_latencia_s: float | None = None

    async def __aenter__(self) -> Cerebro:
        await self.iniciar()
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.fechar()

    async def iniciar(self) -> None:
        if self._cliente is not None:
            return
        remover_chaves_de_api()
        cliente = self._criar_cliente(self._criar_opcoes(self.modelo))
        inicio = time.perf_counter()
        await cliente.connect()
        self._cliente = cliente
        log.info("Conversa aberta (modelo %s) em %.1f s", self.modelo, time.perf_counter() - inicio)

    async def fechar(self) -> None:
        cliente, self._cliente = self._cliente, None
        if cliente is not None:
            try:
                await cliente.disconnect()
            except Exception as erro:  # fechar nunca deve derrubar o assistente
                log.debug("erro ao fechar a conversa: %s", erro)

    async def esquecer(self) -> None:
        """Começa uma conversa nova (sem histórico)."""
        await self.fechar()
        await self.iniciar()

    async def responder(self, pergunta: str) -> AsyncIterator[str]:
        inicio = time.perf_counter()
        self.ultima_latencia_s = None
        entregou = False
        try:
            if self._cliente is None:
                await self.iniciar()
            async for pedaco in self._pedacos(pergunta, inicio):
                if not entregou:
                    self.ultima_latencia_s = time.perf_counter() - inicio
                    log.info("primeiro trecho da resposta em %.2f s", self.ultima_latencia_s)
                entregou = True
                yield pedaco
        except FalhaDoCerebro as falha:
            tipo = falha.tipo
            if tipo == "timeout" and not self._checar_internet():
                tipo = "sem_internet"
            log.warning("falha ao responder (%s): %s", tipo, falha)
            await self._descartar_cliente(interromper=True)
            if not entregou:
                yield FALAS_DE_ERRO[tipo]
        except Exception as erro:  # erros do SDK/processo do Claude viram fala
            tipo = _classificar(erro)
            if tipo == "erro" and not self._checar_internet():
                tipo = "sem_internet"
            log.warning("falha ao responder (%s): %s", tipo, erro)
            await self._descartar_cliente(interromper=False)
            if not entregou:
                yield FALAS_DE_ERRO[tipo]

    async def _descartar_cliente(self, interromper: bool) -> None:
        if interromper and self._cliente is not None:
            try:
                await asyncio.wait_for(self._cliente.interrupt(), timeout=3)
            except Exception as erro:
                log.debug("não consegui interromper: %s", erro)
        await self.fechar()  # a próxima pergunta reconecta

    async def _pedacos(self, pergunta: str, inicio: float) -> AsyncIterator[str]:
        from claude_agent_sdk import AssistantMessage, ResultMessage, StreamEvent, TextBlock

        await self._cliente.query(pergunta)
        mensagens = self._cliente.receive_response().__aiter__()
        teve_delta = False
        entregou = False
        while True:
            limite = self.timeout_total_s if entregou else self.timeout_primeiro_s
            restante = limite - (time.perf_counter() - inicio)
            if restante <= 0:
                raise FalhaDoCerebro("timeout", f"sem resposta em {limite:.0f} s")
            try:
                mensagem = await asyncio.wait_for(mensagens.__anext__(), timeout=restante)
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError:
                raise FalhaDoCerebro("timeout", f"sem resposta em {limite:.0f} s") from None

            if isinstance(mensagem, StreamEvent):
                evento = mensagem.event
                delta = evento.get("delta") or {}
                if evento.get("type") == "content_block_delta" and delta.get("type") == "text_delta":
                    texto = delta.get("text", "")
                    if texto:
                        teve_delta = entregou = True
                        yield texto
            elif isinstance(mensagem, AssistantMessage):
                if mensagem.error:
                    raise FalhaDoCerebro(_ERRO_DO_SDK.get(mensagem.error, "erro"), mensagem.error)
                if not teve_delta:  # sem streaming parcial: usa o texto completo
                    for bloco in mensagem.content:
                        if isinstance(bloco, TextBlock) and bloco.text:
                            entregou = True
                            yield bloco.text
                teve_delta = False
            elif isinstance(mensagem, ResultMessage):
                if mensagem.is_error and not entregou:
                    raise FalhaDoCerebro(_classificar(Exception(mensagem.result or mensagem.subtype)),
                                         mensagem.result or mensagem.subtype)
                return


def criar_cerebro(config) -> Cerebro:
    return Cerebro(config.brain_model, config.brain_timeout_first_s, config.brain_timeout_total_s)


# ---------------------------------------------------------------- frases para o TTS

_FIM_DE_FRASE = re.compile(r'[.!?…]+["\')\]]*(?=\s)|\n+')
_PAUSA = re.compile(r"[,;:](?=\s)")
_MARKDOWN = re.compile(r"[*_#`>]+|^\s*[-•]\s+", re.MULTILINE)


def limpar_para_fala(texto: str) -> str:
    """Tira restos de markdown e espaços repetidos (o TTS leria os símbolos)."""
    return re.sub(r"\s+", " ", _MARKDOWN.sub("", texto).replace("|", " ")).strip()


async def frases(pedacos: AsyncIterator[str], minimo: int = 12, maximo: int = 220) -> AsyncIterator[str]:
    """Junta os pedaços do streaming e entrega frases completas, limpas para falar.

    Frases muito curtas ("Claro.") esperam a próxima para não picotar a fala; trechos longos sem
    ponto final são cortados na última vírgula para o TTS começar logo.
    """
    buffer = ""
    async for pedaco in pedacos:
        buffer += pedaco
        while True:
            corte = None
            for fim in _FIM_DE_FRASE.finditer(buffer):
                if len(limpar_para_fala(buffer[:fim.end()])) >= minimo:
                    corte = fim.end()
                    break
            if corte is None and len(buffer) > maximo:
                pausas = [p.end() for p in _PAUSA.finditer(buffer, 0, maximo)]
                corte = pausas[-1] if pausas else None
            if corte is None:
                break
            frase, buffer = limpar_para_fala(buffer[:corte]), buffer[corte:]
            if frase:
                yield frase
    resto = limpar_para_fala(buffer)
    if resto:
        yield resto


# ---------------------------------------------------------------- comando

async def _conversar(config, perguntas: list[str], falante=None) -> int:
    async with criar_cerebro(config) as cerebro:
        interativo = not perguntas
        if interativo:
            print("Converse digitando (linha vazia para sair).", flush=True)
        while True:
            if interativo:
                pergunta = (await asyncio.to_thread(sys.stdin.readline)).strip()
                if not pergunta:
                    return 0
            elif perguntas:
                pergunta = perguntas.pop(0)
                print(f"> {pergunta}", flush=True)
            else:
                return 0
            inicio = time.perf_counter()
            async for frase in frases(cerebro.responder(pergunta)):
                print(f"  [{time.perf_counter() - inicio:5.2f} s] {frase}", flush=True)
                if falante is not None:
                    falante.falar(frase)  # começa a falar enquanto o resto ainda chega
            if falante is not None:
                await asyncio.to_thread(falante.esperar)


def comando_perguntar(config, perguntas: list[str], falar: bool = False) -> int:
    falante = None
    if falar:
        from chubasa.tts import criar_falante

        falante = criar_falante(config)
    try:
        return asyncio.run(_conversar(config, list(perguntas), falante))
    finally:
        if falante is not None:
            falante.fechar()
