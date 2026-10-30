from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, StreamEvent, TextBlock

from chubasa import brain
from chubasa.config import ConfiguracaoInvalida, carregar_config
from chubasa.tools import Ferramentas, NOME_SERVIDOR
from chubasa.tools.hora import descrever_agora
from chubasa.tools.programas import ErroPrograma, ProgramasPermitidos
from chubasa.tools.timers import ErroTimer, Timers, falar_duracao


def rodar(coro):
    return asyncio.run(coro)


def texto(resultado):
    return resultado["content"][0]["text"]


@pytest.fixture
def avisos():
    lista = []
    evento = threading.Event()

    def avisar(t):
        lista.append(t)
        evento.set()

    avisar.lista = lista
    avisar.evento = evento
    return avisar


@pytest.fixture
def ferramentas(avisos):
    abertos = []
    programas = ProgramasPermitidos.do_texto("calculadora:calc.exe;bloco de notas:notepad.exe", abertos.append)
    agora = datetime(2026, 10, 30, 14, 5, tzinfo=timezone(timedelta(hours=-3), "BRT"))
    f = Ferramentas(avisos, programas, relogio=lambda: agora)
    f.abertos = abertos
    yield f
    f.fechar()


def por_nome(ferramentas, nome):
    return next(d for d in ferramentas.definicoes if d.name == nome)


# ---------------------------------------------------------------- hora

def test_hora_por_extenso():
    agora = datetime(2026, 10, 30, 14, 5, tzinfo=timezone(timedelta(hours=-3), "BRT"))
    assert descrever_agora(agora) == ("Hoje é sexta-feira, 30 de outubro de 2026. "
                                      "Agora são 14 horas e 5 minutos (14:05, fuso BRT).")


def test_hora_em_ponto():
    assert "9 horas em ponto" in descrever_agora(datetime(2026, 1, 5, 9, 0).astimezone())


def test_ferramenta_hora(ferramentas):
    assert "30 de outubro" in texto(rodar(por_nome(ferramentas, "hora_data").handler({})))


# ---------------------------------------------------------------- timers

@pytest.mark.parametrize("segundos, falado", [(300, "5 minutos"), (60, "1 minuto"), (45, "45 segundos"),
                                              (5400, "1 hora e 30 minutos"), (7200, "2 horas"),
                                              (90, "1 minuto e 30 segundos")])
def test_falar_duracao(segundos, falado):
    assert falar_duracao(segundos) == falado


def test_timer_avisa_quando_acaba(avisos):
    timers = Timers(avisos)
    timers.criar(1)
    assert avisos.evento.wait(3)
    assert avisos.lista == ["O timer de 1 segundo acabou."]
    assert timers.ativos() == []


def test_lembrete_usa_o_texto(avisos):
    timers = Timers(avisos)
    timers.criar(1, "tirar o bolo do forno")
    assert avisos.evento.wait(3)
    assert avisos.lista == ["Lembrete: tirar o bolo do forno."]


def test_cancelar_timer_nao_avisa(avisos):
    timers = Timers(avisos)
    timer = timers.criar(1)
    assert timers.cancelar(timer.id)
    time.sleep(1.3)
    assert avisos.lista == []
    assert not timers.cancelar(timer.id)


def test_timer_nao_trava_quem_criou(avisos):
    timers = Timers(avisos)
    inicio = time.perf_counter()
    timers.criar(60)
    assert time.perf_counter() - inicio < 0.1
    timers.cancelar_todos()


@pytest.mark.parametrize("segundos", [0, -5, 24 * 3600 + 1])
def test_timer_com_duracao_invalida(avisos, segundos):
    with pytest.raises(ErroTimer):
        Timers(avisos).criar(segundos)


def test_limite_de_timers_ativos(avisos):
    timers = Timers(avisos)
    for _ in range(10):
        timers.criar(600)
    with pytest.raises(ErroTimer, match="10"):
        timers.criar(600)
    timers.cancelar_todos()


def test_aviso_que_falha_nao_quebra(caplog):
    def avisar(_t):
        raise RuntimeError("placa de som sumiu")

    timers = Timers(avisar)
    timers.criar(1)
    time.sleep(1.4)
    assert "não consegui avisar" in caplog.text


def test_ferramentas_de_timer(ferramentas):
    criado = texto(rodar(por_nome(ferramentas, "criar_timer").handler({"duracao_segundos": 300})))
    assert criado == "Timer 1 criado: aviso em 5 minutos."
    listado = texto(rodar(por_nome(ferramentas, "listar_timers").handler({})))
    assert "Timer 1" in listado and "faltam" in listado
    assert "cancelado" in texto(rodar(por_nome(ferramentas, "cancelar_timer").handler({"id": 1})))
    assert rodar(por_nome(ferramentas, "cancelar_timer").handler({"id": 1})).get("is_error")
    assert texto(rodar(por_nome(ferramentas, "listar_timers").handler({}))) == "Nenhum timer ativo."


def test_ferramenta_timer_invalido_devolve_erro(ferramentas):
    resultado = rodar(por_nome(ferramentas, "criar_timer").handler({"duracao_segundos": 0}))
    assert resultado.get("is_error")


# ---------------------------------------------------------------- programas

def test_abre_so_programa_da_lista():
    abertos = []
    programas = ProgramasPermitidos.do_texto("calculadora:calc.exe;bloco de notas:notepad.exe", abertos.append)

    assert programas.abrir("Calculadora") == "calculadora"
    assert programas.abrir("bloco de  notas") == "bloco de notas"
    assert abertos == ["calc.exe", "notepad.exe"]
    for pedido in ("cmd", "C:\\Windows\\System32\\cmd.exe", "calc.exe", "powershell -c del *"):
        with pytest.raises(ErroPrograma, match="não está na lista"):
            programas.abrir(pedido)
    assert abertos == ["calc.exe", "notepad.exe"]


def test_falha_ao_abrir_vira_erro_claro():
    def falha(_comando):
        raise FileNotFoundError("não encontrado")

    with pytest.raises(ErroPrograma, match="Não consegui abrir"):
        ProgramasPermitidos({"chrome": "chrome.exe"}, falha).abrir("chrome")


@pytest.mark.parametrize("texto_env", ["calculadora", "calculadora:", ":calc.exe"])
def test_lista_mal_formatada(texto_env):
    with pytest.raises(ValueError, match="nome:comando"):
        ProgramasPermitidos.do_texto(texto_env)


def test_lista_invalida_no_env_da_erro_de_config(env_minimo, monkeypatch):
    monkeypatch.setenv("TOOLS_PROGRAMS", "calculadora")
    with pytest.raises(ConfiguracaoInvalida, match="TOOLS_PROGRAMS"):
        carregar_config()


def test_ferramenta_programa(ferramentas):
    assert texto(rodar(por_nome(ferramentas, "abrir_programa").handler({"nome": "calculadora"}))) == "Abri calculadora."
    assert rodar(por_nome(ferramentas, "abrir_programa").handler({"nome": "regedit"})).get("is_error")
    assert ferramentas.abertos == ["calc.exe"]


# ---------------------------------------------------------------- allowlist e integração

def test_allowlist_tem_so_as_ferramentas_do_assistente(ferramentas):
    opcoes = brain.criar_opcoes("sonnet", ferramentas)
    assert opcoes.allowed_tools == [f"mcp__{NOME_SERVIDOR}__{n}" for n in
                                    ("hora_data", "criar_timer", "listar_timers", "cancelar_timer", "abrir_programa")]
    assert opcoes.tools == []
    assert list(opcoes.mcp_servers) == [NOME_SERVIDOR]
    assert opcoes.strict_mcp_config is True
    assert "timers" in opcoes.system_prompt


def test_sem_ferramentas_prompt_avisa_que_nao_tem():
    opcoes = brain.criar_opcoes("sonnet", None)
    assert opcoes.allowed_tools == [] and opcoes.mcp_servers == {}
    assert "não tem acesso a internet, arquivos, relógio" in opcoes.system_prompt


class ClienteQueUsaFerramenta:
    """Faz o papel do CLI: fala com o servidor MCP real pelo mesmo bridge que o SDK usa."""

    def __init__(self, opcoes):
        self.opcoes = opcoes
        self.lista_de_ferramentas = None

    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def query(self, pergunta):
        self.pergunta = pergunta

    async def _mcp(self, bridge, id_, metodo, params):
        return await bridge.handle({"jsonrpc": "2.0", "id": id_, "method": metodo, "params": params})

    async def receive_response(self):
        from claude_agent_sdk._internal.sdk_mcp_bridge import SdkMcpBridge

        servidor = self.opcoes.mcp_servers[NOME_SERVIDOR]["instance"]
        bridge = SdkMcpBridge(NOME_SERVIDOR, servidor)
        await self._mcp(bridge, 1, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                                    "clientInfo": {"name": "teste", "version": "1"}})
        await bridge.handle({"jsonrpc": "2.0", "method": "notifications/initialized"})
        lista = await self._mcp(bridge, 2, "tools/list", {})
        self.lista_de_ferramentas = [t["name"] for t in lista["result"]["tools"]]
        # "coloca um timer de 5 minutos" -> o modelo chamaria criar_timer(300)
        chamada = await self._mcp(bridge, 3, "tools/call", {"name": "criar_timer",
                                                            "arguments": {"duracao_segundos": 300}})
        self.resultado = chamada["result"]["content"][0]["text"]
        await bridge.aclose()
        yield StreamEvent(uuid="u", session_id="s", event={
            "type": "content_block_delta", "delta": {"type": "text_delta", "text": "Pronto, timer de cinco minutos."}})
        yield AssistantMessage(content=[TextBlock(text="Pronto, timer de cinco minutos.")], model="sonnet")
        yield ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1, is_error=False, num_turns=2,
                            session_id="s")


def test_coloca_um_timer_de_5_minutos_pelo_cerebro(ferramentas):
    clientes = []

    def criar(opcoes):
        clientes.append(ClienteQueUsaFerramenta(opcoes))
        return clientes[0]

    cerebro = brain.Cerebro("sonnet", criar_cliente=criar,
                            criar_opcoes_fn=lambda m: brain.criar_opcoes(m, ferramentas))

    async def fluxo():
        return [p async for p in brain.frases(cerebro.responder("coloca um timer de 5 minutos"))]

    falado = rodar(fluxo())

    assert falado == ["Pronto, timer de cinco minutos."]
    assert clientes[0].lista_de_ferramentas == ["hora_data", "criar_timer", "listar_timers", "cancelar_timer",
                                                "abrir_programa"]
    assert clientes[0].resultado == "Timer 1 criado: aviso em 5 minutos."
    [timer] = ferramentas.timers.ativos()
    assert timer.duracao_s == 300 and 295 < timer.restante_s() <= 300
