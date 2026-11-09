"""Ferramentas que o assistente pode usar na conversa, como servidor MCP dentro do próprio processo.

Só as ferramentas listadas aqui ficam liberadas (allowlist). Nenhuma apaga arquivo nem envia
mensagem: hora/data, timers/lembretes e abrir programas de uma lista fixa do `.env`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from chubasa.tools.atalhos import AtalhosDoMenuIniciar, ProgramaAmbiguo
from chubasa.tools.hora import descrever_agora
from chubasa.tools.midia import ACOES_MIDIA, ACOES_VOLUME, MAXIMO_PASSOS, ErroMidia, TeclasDeMidia
from chubasa.tools.navegador import ErroNavegador, Navegador
from chubasa.tools.programas import ErroPrograma, ProgramasPermitidos
from chubasa.tools.timers import ErroTimer, Timers, falar_duracao

NOME_SERVIDOR = "chubasa"

log = logging.getLogger(__name__)


def _texto(texto: str, erro: bool = False) -> dict[str, Any]:
    resultado: dict[str, Any] = {"content": [{"type": "text", "text": texto}]}
    if erro:
        resultado["is_error"] = True
    return resultado


class Ferramentas:
    """Junta o estado das ferramentas (timers, programas) e monta o servidor MCP in-process."""

    def __init__(self, avisar: Callable[[str], None], programas: ProgramasPermitidos,
                 relogio: Callable[[], Any] | None = None, timers: Timers | None = None,
                 navegador: Navegador | None = None, teclas: TeclasDeMidia | None = None):
        self.timers = timers if timers is not None else Timers(avisar)
        self.programas = programas
        self.navegador = navegador if navegador is not None else Navegador()
        self.teclas = teclas if teclas is not None else TeclasDeMidia()
        self._relogio = relogio
        self.definicoes = self._definir()

    @property
    def nomes_permitidos(self) -> list[str]:
        """Nomes completos para `allowed_tools` (prefixo mcp__servidor__)."""
        return [f"mcp__{NOME_SERVIDOR}__{d.name}" for d in self.definicoes]

    def servidor(self) -> Any:
        from claude_agent_sdk import create_sdk_mcp_server

        return create_sdk_mcp_server(name=NOME_SERVIDOR, version="1.0.0", tools=self.definicoes)

    def fechar(self) -> None:
        self.timers.cancelar_todos()

    def _definir(self) -> list[Any]:
        from claude_agent_sdk import tool

        programas = self.programas
        navegador = self.navegador
        teclas = self.teclas
        timers = self.timers
        relogio = self._relogio

        @tool("hora_data", "Diz a data e a hora atuais do computador (fuso local).", {})
        async def hora_data(_args: dict[str, Any]) -> dict[str, Any]:
            return _texto(descrever_agora(relogio() if relogio else None))

        @tool(
            "criar_timer",
            "Cria um timer ou lembrete. Quando o tempo acabar, o assistente avisa em voz alta. "
            "duracao_segundos: tempo até o aviso (ex.: 5 minutos = 300). "
            "lembrete: o que dizer no aviso (opcional, ex.: 'tirar o bolo do forno').",
            {
                "type": "object",
                "properties": {
                    "duracao_segundos": {"type": "integer", "minimum": 1},
                    "lembrete": {"type": "string"},
                },
                "required": ["duracao_segundos"],
            },
        )
        async def criar_timer(args: dict[str, Any]) -> dict[str, Any]:
            try:
                timer = timers.criar(int(args["duracao_segundos"]), str(args.get("lembrete") or ""))
            except (ErroTimer, ValueError, TypeError) as erro:
                return _texto(f"Não criei o timer: {erro}", erro=True)
            return _texto(f"Timer {timer.id} criado: aviso em {falar_duracao(timer.duracao_s)}.")

        @tool("listar_timers", "Lista os timers e lembretes ativos e quanto falta para cada um.", {})
        async def listar_timers(_args: dict[str, Any]) -> dict[str, Any]:
            ativos = timers.ativos()
            if not ativos:
                return _texto("Nenhum timer ativo.")
            linhas = [f"Timer {t.id}{' (' + t.lembrete + ')' if t.lembrete else ''}: faltam "
                      f"{falar_duracao(t.restante_s())}." for t in ativos]
            return _texto(" ".join(linhas))

        @tool("cancelar_timer", "Cancela um timer pelo número (use listar_timers para ver os números).",
              {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]})
        async def cancelar_timer(args: dict[str, Any]) -> dict[str, Any]:
            if timers.cancelar(int(args["id"])):
                return _texto(f"Timer {args['id']} cancelado.")
            return _texto(f"Não existe timer {args['id']} ativo.", erro=True)

        @tool(
            "abrir_programa",
            "Abre um programa do computador pelo nome. Atalhos fixos: " + ", ".join(programas.nomes())
            + (". Também abre qualquer programa do Menu Iniciar do Windows pelo nome (ex.: "
               "'Spotify', 'Word', 'Paint')" if programas.atalhos is not None else "")
            + ". Passe só o nome do programa, nunca um caminho. Os nomes do Menu Iniciar podem estar "
              "em inglês: se não achar, tente o nome em inglês (ex.: 'mapa de caracteres' -> "
              "'Character Map'). Se a resposta trouxer opções, pergunte à pessoa qual delas.",
            {"type": "object", "properties": {"nome": {"type": "string"}}, "required": ["nome"]},
        )
        async def abrir_programa(args: dict[str, Any]) -> dict[str, Any]:
            try:
                nome = programas.abrir(str(args.get("nome", "")))
            except ProgramaAmbiguo as ambiguo:
                return _texto(f"Encontrei mais de um programa para '{ambiguo.pedido}': "
                              f"{', '.join(ambiguo.opcoes)}. Pergunte qual deles abrir.")
            except ErroPrograma as erro:
                return _texto(str(erro), erro=True)
            return _texto(f"Abri {nome}.")

        def no_navegador(acao: Callable[[str], str], valor: str, feito: str) -> dict[str, Any]:
            try:
                acao(valor)
            except ErroNavegador as erro:
                return _texto(f"Não abri: {erro}.", erro=True)
            return _texto(feito)

        @tool("abrir_site",
              "Abre um site no navegador padrão. url: endereço http/https ou só o domínio "
              "(ex.: 'wikipedia.org'). Use quando pedirem para abrir ou mostrar um site.",
              {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]})
        async def abrir_site(args: dict[str, Any]) -> dict[str, Any]:
            return no_navegador(navegador.abrir_site, str(args.get("url", "")), "Abri o site no navegador.")

        @tool("pesquisar_no_google",
              "Abre no navegador a página de resultados do Google para o termo. Use quando pedirem "
              "para pesquisar ou procurar algo NO NAVEGADOR; para só responder uma pergunta, use a "
              "pesquisa na internet.",
              {"type": "object", "properties": {"termo": {"type": "string"}}, "required": ["termo"]})
        async def pesquisar_no_google(args: dict[str, Any]) -> dict[str, Any]:
            return no_navegador(navegador.pesquisar_no_google, str(args.get("termo", "")),
                                "Abri a pesquisa do Google no navegador.")

        @tool("pesquisar_no_youtube",
              "Abre no navegador a busca do YouTube para o termo (vídeos, músicas, clipes).",
              {"type": "object", "properties": {"termo": {"type": "string"}}, "required": ["termo"]})
        async def pesquisar_no_youtube(args: dict[str, Any]) -> dict[str, Any]:
            return no_navegador(navegador.pesquisar_no_youtube, str(args.get("termo", "")),
                                "Abri a busca do YouTube no navegador.")

        @tool("midia",
              "Controla o que está tocando no computador (Spotify, YouTube etc.) pelas teclas de mídia. "
              "acao: tocar_pausar, proxima, anterior ou parar.",
              {"type": "object", "properties": {"acao": {"type": "string", "enum": list(ACOES_MIDIA)}},
               "required": ["acao"]})
        async def midia(args: dict[str, Any]) -> dict[str, Any]:
            try:
                acao = teclas.midia(str(args.get("acao", "")))
            except ErroMidia as erro:
                return _texto(str(erro), erro=True)
            return _texto(f"Mídia: {acao.replace('_', '/')} feito.")

        @tool("volume",
              "Muda o volume do computador. acao: aumentar, diminuir ou mudo (liga/desliga o mudo). "
              f"passos: quantos passos de 2% (1 a {MAXIMO_PASSOS}; 'um pouco' = 5).",
              {"type": "object", "properties": {
                  "acao": {"type": "string", "enum": list(ACOES_VOLUME)},
                  "passos": {"type": "integer", "minimum": 1, "maximum": MAXIMO_PASSOS}},
               "required": ["acao"]})
        async def volume(args: dict[str, Any]) -> dict[str, Any]:
            try:
                passos = teclas.volume(str(args.get("acao", "")), int(args.get("passos") or 5))
            except (ErroMidia, ValueError, TypeError) as erro:
                return _texto(str(erro), erro=True)
            if passos == 0:
                return _texto("Mudo ligado ou desligado.")
            return _texto(f"Volume: {args['acao']} {passos * 2}%.")

        return [hora_data, criar_timer, listar_timers, cancelar_timer, abrir_programa,
                abrir_site, pesquisar_no_google, pesquisar_no_youtube, midia, volume]


def criar_ferramentas(config, avisar: Callable[[str], None]) -> Ferramentas:
    import sys

    atalhos = AtalhosDoMenuIniciar() if sys.platform == "win32" else None
    return Ferramentas(avisar, ProgramasPermitidos.do_texto(config.tools_programs, atalhos=atalhos))
