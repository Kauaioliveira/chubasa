from __future__ import annotations

import asyncio
import os

import pytest
from claude_agent_sdk import AssistantMessage, CLIConnectionError, ResultMessage, StreamEvent, TextBlock

from chubasa import brain
from chubasa.config import RAIZ


def delta(texto):
    return StreamEvent(uuid="u", session_id="s",
                       event={"type": "content_block_delta", "index": 0,
                              "delta": {"type": "text_delta", "text": texto}})


def assistente(texto, erro=None):
    return AssistantMessage(content=[TextBlock(text=texto)] if texto else [], model="sonnet", error=erro)


def resultado(erro=False, texto=None):
    return ResultMessage(subtype="error_during_execution" if erro else "success", duration_ms=1,
                         duration_api_ms=1, is_error=erro, num_turns=1, session_id="s", result=texto)


class ClienteFalso:
    """Imita o ClaudeSDKClient: cada pergunta recebe a próxima lista de mensagens do roteiro."""

    def __init__(self, roteiro, erro_na_query=None, atraso=0.0):
        self.roteiro = list(roteiro)
        self.erro_na_query = erro_na_query
        self.atraso = atraso
        self.perguntas = []
        self.conectado = False
        self.interrompido = False

    async def connect(self):
        self.conectado = True

    async def disconnect(self):
        self.conectado = False

    async def interrupt(self):
        self.interrompido = True

    async def query(self, texto):
        if self.erro_na_query:
            raise self.erro_na_query
        self.perguntas.append(texto)

    async def receive_response(self):
        for mensagem in self.roteiro.pop(0):
            if self.atraso:
                await asyncio.sleep(self.atraso)
            yield mensagem


def cerebro_com(*clientes, internet=True, **kwargs):
    fila = list(clientes)
    criados = []

    def criar(_opcoes):
        cliente = fila.pop(0)
        criados.append(cliente)
        return cliente

    c = brain.Cerebro("sonnet", criar_cliente=criar, criar_opcoes_fn=lambda m: {"model": m},
                      checar_internet=lambda: internet, **kwargs)
    return c, criados


async def juntar(gerador):
    return [p async for p in gerador]


def rodar(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------- streaming e histórico

def test_streaming_entrega_pedacos_sem_duplicar_a_mensagem_final():
    cliente = ClienteFalso([[delta("Olá, "), delta("tudo bem?"), assistente("Olá, tudo bem?"), resultado()]])
    cerebro, _ = cerebro_com(cliente)

    pedacos = rodar(juntar(cerebro.responder("oi")))

    assert pedacos == ["Olá, ", "tudo bem?"]
    assert cerebro.ultima_latencia_s is not None
    assert cliente.perguntas == ["oi"]


def test_sem_streaming_usa_o_texto_completo():
    cerebro, _ = cerebro_com(ClienteFalso([[assistente("Resposta inteira."), resultado()]]))
    assert rodar(juntar(cerebro.responder("oi"))) == ["Resposta inteira."]


def test_mesmo_cliente_entre_perguntas():
    cliente = ClienteFalso([[delta("Sete."), resultado()], [delta("Vinte e um."), resultado()]])
    cerebro, criados = cerebro_com(cliente)

    async def conversa():
        await juntar(cerebro.responder("pensa num número"))
        await juntar(cerebro.responder("multiplica por três"))

    rodar(conversa())
    assert len(criados) == 1
    assert cliente.perguntas == ["pensa num número", "multiplica por três"]


def test_esquecer_abre_conversa_nova():
    cerebro, criados = cerebro_com(ClienteFalso([]), ClienteFalso([]))

    async def fluxo():
        await cerebro.iniciar()
        await cerebro.esquecer()

    rodar(fluxo())
    assert len(criados) == 2
    assert not criados[0].conectado and criados[1].conectado


# ---------------------------------------------------------------- erros viram fala

@pytest.mark.parametrize("erro_sdk, tipo", [("rate_limit", "limite"), ("billing_error", "limite"),
                                            ("authentication_failed", "autenticacao"),
                                            ("server_error", "erro")])
def test_erro_do_modelo_vira_frase_e_reconecta(erro_sdk, tipo):
    primeiro = ClienteFalso([[assistente("", erro=erro_sdk), resultado(erro=True)]])
    segundo = ClienteFalso([[delta("Voltei."), resultado()]])
    cerebro, criados = cerebro_com(primeiro, segundo)

    async def fluxo():
        return await juntar(cerebro.responder("oi")), await juntar(cerebro.responder("e agora?"))

    falha, depois = rodar(fluxo())
    assert falha == [brain.FALAS_DE_ERRO[tipo]]
    assert depois == ["Voltei."]
    assert len(criados) == 2


def test_falha_de_conexao_sem_internet():
    cliente = ClienteFalso([], erro_na_query=CLIConnectionError("processo morreu"))
    cerebro, _ = cerebro_com(cliente, internet=False)
    assert rodar(juntar(cerebro.responder("oi"))) == [brain.FALAS_DE_ERRO["sem_internet"]]


def test_resultado_com_erro_de_limite():
    cliente = ClienteFalso([[resultado(erro=True, texto="Claude usage limit reached")]])
    cerebro, _ = cerebro_com(cliente)
    assert rodar(juntar(cerebro.responder("oi"))) == [brain.FALAS_DE_ERRO["limite"]]


def test_timeout_interrompe_e_avisa():
    cliente = ClienteFalso([[delta("tarde demais"), resultado()]], atraso=0.5)
    cerebro, _ = cerebro_com(cliente, timeout_primeiro_s=0.1)

    assert rodar(juntar(cerebro.responder("oi"))) == [brain.FALAS_DE_ERRO["timeout"]]
    assert cliente.interrompido


def test_timeout_sem_internet_vira_sem_internet():
    cliente = ClienteFalso([[delta("x")]], atraso=0.5)
    cerebro, _ = cerebro_com(cliente, internet=False, timeout_primeiro_s=0.1)
    assert rodar(juntar(cerebro.responder("oi"))) == [brain.FALAS_DE_ERRO["sem_internet"]]


def test_erro_depois_de_comecar_nao_repete_frase_de_erro():
    cliente = ClienteFalso([[delta("Começo da resposta. "), assistente("", erro="server_error")]])
    cerebro, _ = cerebro_com(cliente)
    assert rodar(juntar(cerebro.responder("oi"))) == ["Começo da resposta. "]


# ---------------------------------------------------------------- isolamento

def test_remove_chaves_do_ambiente(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-teste")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://proxy")
    assert sorted(brain.remover_chaves_de_api()) == ["ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"]
    assert "ANTHROPIC_API_KEY" not in os.environ


def test_iniciar_remove_chaves_antes_de_criar_o_cliente(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-teste")
    visto = []

    def criar(_opcoes):
        visto.append("ANTHROPIC_API_KEY" in os.environ)
        return ClienteFalso([])

    rodar(brain.Cerebro(criar_cliente=criar, criar_opcoes_fn=lambda m: m).iniciar())
    assert visto == [False]


def test_opcoes_sem_ferramentas_sem_settings_e_com_streaming(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    opcoes = brain.criar_opcoes("haiku")

    assert opcoes.model == "haiku"
    assert opcoes.tools == []
    assert opcoes.allowed_tools == []
    assert {"Bash", "Read", "Write", "Edit", "WebFetch"} <= set(opcoes.disallowed_tools)
    assert opcoes.setting_sources == []
    assert opcoes.strict_mcp_config is True and opcoes.mcp_servers == {}
    assert opcoes.include_partial_messages is True
    assert RAIZ not in opcoes.cwd.parents  # longe do CLAUDE.md do projeto
    assert "Chubasa" in opcoes.system_prompt and "markdown" in opcoes.system_prompt


# ---------------------------------------------------------------- frases

async def gerar(*pedacos):
    for p in pedacos:
        yield p


def frases_de(*pedacos, **kwargs):
    return rodar(juntar(brain.frases(gerar(*pedacos), **kwargs)))


def test_frases_cortam_no_ponto_mesmo_em_pedacos_quebrados():
    assert frases_de("Hoje vai fazer sol em São", " Paulo. Amanhã pode cho", "ver à tarde! E depois?") == [
        "Hoje vai fazer sol em São Paulo.", "Amanhã pode chover à tarde!", "E depois?"]


def test_frase_curta_espera_a_proxima():
    assert frases_de("Claro. ", "O número que eu pensei foi sete. ") == ["Claro. O número que eu pensei foi sete."]


def test_numero_decimal_nao_e_fim_de_frase():
    assert frases_de("A temperatura é 3.5 graus agora. ") == ["A temperatura é 3.5 graus agora."]


def test_limpa_markdown():
    assert frases_de("**Resposta:** use o `comando`.\n- item um\n") == ["Resposta: use o comando.", "item um"]


def test_texto_longo_sem_ponto_corta_na_virgula():
    longo = "uma parte bem comprida da resposta, " * 8
    saida = frases_de(longo, maximo=100)
    assert len(saida) > 1
    assert all(len(f) <= 110 for f in saida[:-1])


def test_sem_texto_nao_entrega_nada():
    assert frases_de("", "  ") == []
