"""Ferramentas que o assistente pode usar na conversa, como servidor MCP dentro do próprio processo.

Só as ferramentas listadas aqui ficam liberadas (allowlist). Nenhuma apaga arquivo nem envia
mensagem: hora/data, timers/lembretes e abrir programas de uma lista fixa do `.env`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from chubasa.tools.hora import descrever_agora
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
                 relogio: Callable[[], Any] | None = None, timers: Timers | None = None):
        self.timers = timers if timers is not None else Timers(avisar)
        self.programas = programas
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
            "Abre um programa do computador. Só funciona com os nomes desta lista: "
            + ", ".join(programas.nomes()) + ".",
            {"type": "object", "properties": {"nome": {"type": "string", "enum": programas.nomes()}},
             "required": ["nome"]},
        )
        async def abrir_programa(args: dict[str, Any]) -> dict[str, Any]:
            try:
                nome = programas.abrir(str(args.get("nome", "")))
            except ErroPrograma as erro:
                return _texto(str(erro), erro=True)
            return _texto(f"Abri {nome}.")

        return [hora_data, criar_timer, listar_timers, cancelar_timer, abrir_programa]


def criar_ferramentas(config, avisar: Callable[[str], None]) -> Ferramentas:
    return Ferramentas(avisar, ProgramasPermitidos.do_texto(config.tools_programs))
