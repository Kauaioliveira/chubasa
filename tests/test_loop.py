from __future__ import annotations

import asyncio
import threading

import numpy as np
import pytest

from chubasa import loop
from chubasa.audio import AMOSTRAS_POR_FRAME, ErroDeAudio
from chubasa.speaker import ResultadoVoz
from chubasa.stt import Transcricao
from chubasa.vad import ResultadoGravacao
from chubasa.wakeword import Deteccao

SILENCIO, PALAVRA, PALAVRA_FORTE = 0, 1, 2


def frame(tipo=SILENCIO):
    return np.full(AMOSTRAS_POR_FRAME, tipo, dtype=np.int16)


class MicFalso:
    """Entrega os frames do roteiro; quando acabam, para o assistente."""

    def __init__(self, frames, assistente_ref, ao_ler=None):
        self.frames = list(frames)
        self.ref = assistente_ref
        self.ao_ler = ao_ler
        self.aberturas = 0
        self.descartes = 0

    def __call__(self):
        self.aberturas += 1
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def ler(self, timeout=1.0):
        if self.ao_ler:
            self.ao_ler()
        if not self.frames:
            self.ref[0].parar()
            raise ErroDeAudio("fim do roteiro")
        return self.frames.pop(0)

    def descartar_pendentes(self):
        self.descartes += 1
        return 0


class DetectorFalso:
    frase = "hey chubasa"

    def __init__(self):
        self.processados = 0
        self.reinicios = 0

    def processar(self, f):
        self.processados += 1
        tipo = int(f[0])
        if tipo == PALAVRA:
            return Deteccao("hey_chubasa", 0.6, 0.0)
        if tipo == PALAVRA_FORTE:
            return Deteccao("hey_chubasa", 0.95, 0.0)
        return None

    def audio_recente(self):
        return np.full(800, 9, dtype=np.int16)

    def reiniciar(self):
        self.reinicios += 1


class GravadorFalso:
    """Cada chamada consome 3 frames e devolve fala ou silêncio conforme o roteiro."""

    def __init__(self, roteiro):
        self.roteiro = list(roteiro)
        self.chamadas = []

    def gravar(self, frames, audio_palavra, timeout_sem_fala_s=None):
        self.chamadas.append((len(audio_palavra), timeout_sem_fala_s))
        for _ in range(3):
            next(frames, None)
        teve = self.roteiro.pop(0) if self.roteiro else False
        if not teve:
            return ResultadoGravacao("sem_fala", audio_palavra, np.zeros(0, np.int16))
        return ResultadoGravacao("fim_de_fala", audio_palavra, np.ones(1600, np.int16))


class VerificadorFalso:
    def __init__(self, ativo=False, aceita=True):
        self.ativo = ativo
        self.aceita = aceita
        self.chamadas = []

    def verificar(self, palavra, comando=None):
        self.chamadas.append((len(palavra), None if comando is None else len(comando)))
        return ResultadoVoz(self.aceita, 0.7 if self.ativo else None, self.ativo)


class TranscritorFalso:
    def __init__(self, textos):
        self.textos = list(textos)

    def transcrever(self, _audio):
        return Transcricao(self.textos.pop(0) if self.textos else "", 1.0, 0.1)


class CerebroFalso:
    timeout_total_s = 5.0

    def __init__(self, respostas=None):
        self.respostas = list(respostas or [])
        self.perguntas = []
        self.esquecimentos = 0
        self.iniciado = self.fechado = False

    async def iniciar(self):
        self.iniciado = True

    async def fechar(self):
        self.fechado = True

    async def esquecer(self):
        self.esquecimentos += 1

    async def responder(self, texto):
        self.perguntas.append(texto)
        for pedaco in (self.respostas.pop(0) if self.respostas else ["Ok. "]):
            await asyncio.sleep(0)
            yield pedaco


class FalanteFalso:
    """Fica 'falando' por `frames_falando` leituras do microfone depois de cada fala."""

    def __init__(self, frames_falando=2):
        self.falando = threading.Event()
        self.falas = []
        self.interrupcoes = 0
        self.frames_falando = frames_falando
        self._restante = 0

    def falar(self, texto):
        self.falas.append(texto)
        self.falando.set()
        self._restante = self.frames_falando

    def tick(self):
        if self._restante > 0:
            self._restante -= 1
            if self._restante == 0:
                self.falando.clear()

    def interromper(self):
        self.interrupcoes += 1
        self._restante = 0
        self.falando.clear()


def montar(frames, gravacoes=(), textos=(), respostas=None, verificador=None, conversa_s=6.0,
           frames_falando=2, **kwargs):
    ref = [None]
    falante = FalanteFalso(frames_falando)
    mic = MicFalso(frames, ref, ao_ler=falante.tick)
    componentes = loop.Componentes(
        detector=DetectorFalso(), gravador=GravadorFalso(gravacoes), verificador=verificador or VerificadorFalso(),
        transcritor=TranscritorFalso(textos), cerebro=CerebroFalso(respostas), falante=falante,
        abrir_microfone=mic, bip=lambda: None,
    )
    estados = []
    assistente = loop.Assistente(componentes, conversa_s=conversa_s, ao_mudar_estado=estados.append, **kwargs)
    ref[0] = assistente
    return assistente, componentes, mic, estados


def rodar(assistente):
    asyncio.run(asyncio.wait_for(assistente.rodar(), timeout=10))


def roteiro(*partes, fim=60):
    frames = []
    for parte in partes:
        frames.extend([frame(SILENCIO)] * 3 if parte == "." else [frame(parte)])
    return frames + [frame(SILENCIO)] * fim


# ---------------------------------------------------------------- fluxo principal

def test_palavra_comando_resposta_e_volta_a_dormir():
    assistente, c, _, estados = montar(
        roteiro(".", PALAVRA, "."), gravacoes=[True, False], textos=["que horas são"],
        respostas=[["Agora são ", "três horas. ", "Mais alguma coisa?"]])

    rodar(assistente)

    assert c.cerebro.perguntas == ["que horas são"]
    assert c.falante.falas == ["Agora são três horas.", "Mais alguma coisa?"]
    assert [e.value for e in estados[:5]] == ["ouvindo", "pensando", "falando", "conversando", "dormindo"]
    assert c.verificador.chamadas[0] == (800, 1600)  # palavra-chave + comando juntos
    assert c.cerebro.iniciado and c.cerebro.fechado
    assert assistente.interacoes == 1


def test_modo_conversa_aceita_pergunta_sem_palavra_chave():
    assistente, c, _, _ = montar(roteiro(PALAVRA), gravacoes=[True, True, False],
                                 textos=["pensa num número", "multiplica por três"], conversa_s=4.0)

    rodar(assistente)

    assert c.cerebro.perguntas == ["pensa num número", "multiplica por três"]
    assert c.gravador.chamadas[1] == (0, 4.0)  # sem palavra-chave, timeout do modo conversa
    assert c.verificador.chamadas[1] == (0, 1600)


def test_modo_conversa_desligado():
    assistente, c, _, _ = montar(roteiro(PALAVRA), gravacoes=[True, True], textos=["a", "b"], conversa_s=0)
    rodar(assistente)
    assert c.cerebro.perguntas == ["a"]
    assert len(c.gravador.chamadas) == 1


def test_ninguem_falou_depois_do_bip():
    assistente, c, _, _ = montar(roteiro(PALAVRA), gravacoes=[False])
    rodar(assistente)
    assert c.cerebro.perguntas == []
    assert c.verificador.chamadas == []


def test_voz_de_outra_pessoa_e_ignorada():
    verificador = VerificadorFalso(ativo=True, aceita=False)
    assistente, c, _, _ = montar(roteiro(PALAVRA), gravacoes=[True], textos=["oi"], verificador=verificador)
    rodar(assistente)
    assert c.cerebro.perguntas == []


def test_transcricao_vazia_nao_pergunta():
    assistente, c, _, _ = montar(roteiro(PALAVRA), gravacoes=[True], textos=[""])
    rodar(assistente)
    assert c.cerebro.perguntas == []


def test_comando_esquece_limpa_historico():
    assistente, c, _, _ = montar(roteiro(PALAVRA), gravacoes=[True], textos=["Esquece."])
    rodar(assistente)
    assert c.cerebro.esquecimentos == 1
    assert c.cerebro.perguntas == []
    assert c.falante.falas == [loop.FALA_ESQUECI]


def test_comando_para_cancela_a_fala():
    assistente, c, _, _ = montar(roteiro(PALAVRA), gravacoes=[True], textos=["Para!"])
    rodar(assistente)
    assert c.falante.interrupcoes >= 1
    assert c.cerebro.perguntas == []


def test_palavra_chave_muda_enquanto_fala_aviso():
    assistente, c, _, _ = montar(roteiro(PALAVRA), gravacoes=[False], frames_falando=0)
    c.falante.falando.set()  # ex.: aviso de timer tocando
    c.falante._restante = 10
    rodar(assistente)
    assert c.detector.processados > 0
    assert c.gravador.chamadas == []  # a "palavra" no meio do aviso não ativou


# ---------------------------------------------------------------- interrupção

def test_interromper_com_palavra_forte_e_voz_verificada():
    verificador = VerificadorFalso(ativo=True, aceita=True)
    frames = roteiro(PALAVRA, fim=0) + [frame(SILENCIO)] * 3 + [frame(PALAVRA_FORTE)] + [frame(SILENCIO)] * 60
    assistente, c, _, _ = montar(frames, gravacoes=[True, True, False], textos=["conta uma história", "para"],
                                 respostas=[["Era uma vez um reino muito distante. "]], verificador=verificador,
                                 frames_falando=50)

    rodar(assistente)

    assert c.falante.interrupcoes >= 1
    assert c.gravador.chamadas[1][0] == 800  # depois de interromper, grava com a palavra-chave


def test_sem_voz_cadastrada_nao_interrompe():
    frames = roteiro(PALAVRA, fim=0) + [frame(PALAVRA_FORTE)] * 3 + [frame(SILENCIO)] * 60
    assistente, c, _, _ = montar(frames, gravacoes=[True, False], textos=["oi"], frames_falando=10)

    assert assistente.interrupcao is False
    rodar(assistente)
    assert c.falante.interrupcoes == 0


def test_palavra_fraca_nao_interrompe():
    verificador = VerificadorFalso(ativo=True)
    frames = roteiro(PALAVRA, fim=0) + [frame(PALAVRA)] * 3 + [frame(SILENCIO)] * 60
    assistente, c, _, _ = montar(frames, gravacoes=[True, False], textos=["oi"], verificador=verificador,
                                 frames_falando=10)
    rodar(assistente)
    assert c.falante.interrupcoes == 0


# ---------------------------------------------------------------- controle

def test_pausado_nao_abre_o_microfone_ate_retomar():
    assistente, c, mic, estados = montar(roteiro("."))
    assistente.pausar()

    async def fluxo():
        tarefa = asyncio.create_task(assistente.rodar())
        await asyncio.sleep(0.3)
        assert mic.aberturas == 0
        assert assistente.estado == loop.Estado.PAUSADO
        assistente.retomar()
        await asyncio.wait_for(tarefa, timeout=5)

    asyncio.run(fluxo())
    assert mic.aberturas == 1


def test_cerebro_que_falha_ao_abrir_nao_derruba():
    assistente, c, _, _ = montar(roteiro("."))

    async def falha():
        raise RuntimeError("sem login")

    c.cerebro.iniciar = falha
    rodar(assistente)
    assert c.cerebro.fechado


@pytest.mark.parametrize("texto, esperado", [
    ("Esquece.", "esquecer"), ("esqueça tudo", "esquecer"), ("Nova conversa!", "esquecer"),
    ("Para.", "parar"), ("pare", "parar"), ("Cancela", "parar"), ("Hey Chubasa, para!", "parar"),
    ("Para onde vai o ônibus?", None), ("Esquece a ideia de chover amanhã?", None), ("Que horas são?", None),
])
def test_comando_de_voz(texto, esperado):
    assert loop.comando_de_voz(texto) == esperado


@pytest.mark.parametrize("texto, esperado", [
    ("Hey Chubasa, que horas são?", "que horas são?"), ("ei chubasa! liga a luz", "liga a luz"),
    ("Hey Jarvis. Qual a capital?", "Qual a capital?"), ("Que horas são?", "Que horas são?"),
    ("Chubasa é um nome legal", "Chubasa é um nome legal"),
])
def test_remove_palavra_chave_do_inicio(texto, esperado):
    assert loop.remover_palavra_chave(texto) == esperado


def test_pergunta_vai_sem_a_palavra_chave():
    assistente, c, _, _ = montar(roteiro(PALAVRA), gravacoes=[True, False], textos=["Hey Chubasa, que horas são?"])
    rodar(assistente)
    assert c.cerebro.perguntas == ["que horas são?"]
