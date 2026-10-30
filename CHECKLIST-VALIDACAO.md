# Checklist de validação com voz real

O que os testes automáticos não conseguem provar: microfone, sua voz, alto-falante e a conta.
Siga na ordem; cada item diz o comando e o critério de pronto do plano. Anote os números do log
(`%LOCALAPPDATA%\Chubasa\logs\chubasa.log`) para calibrar o `.env`.

## 0. Preparação (uma vez)

- [ ] `python -m pip install -e ".[voz]"`, `Copy-Item .env.example .env`, `python -m chubasa config`
      imprime **"configuração ok"**.
- [ ] Login: `claude` → `/login` com a conta do plano Pro → `/exit`.
- [ ] Nenhuma `ANTHROPIC_API_KEY` no sistema (o log avisaria em destaque).

## 1. Microfone (Bloco 1)

- [ ] `python -m chubasa dispositivos` lista o seu microfone (marcado com `*` se for o padrão).
- [ ] `python -m chubasa medidor`: a barra sobe quando você fala e cai em silêncio.
- [ ] `python -m chubasa gravar-teste` e ouça `teste.wav`.
      **Pronto quando:** o `teste.wav` toca a sua voz limpa.

## 2. Palavra-chave (Bloco 2)

Com o modelo pronto, diga "hey jarvis"; depois do treino (treino/LEIA-ME.md), "hey chubasa".

- [ ] `python -m chubasa escutar`: diga a palavra-chave 10 vezes, em tom normal, a 1–2 m do microfone.
      **Pronto quando:** detecta pelo menos **9 de 10** (cada detecção toca um bip).
- [ ] Deixe o `escutar` ligado por **10 minutos** de conversa normal ou TV.
      **Pronto quando:** no máximo **1** falso disparo.
- [ ] Se errar: ajuste `WAKE_THRESHOLD` olhando "score máximo" no log (desça se não ouve, suba se dispara à toa).

## 3. Verificação de voz (Bloco 2.5)

- [ ] `python -m chubasa cadastrar-voz` (15 frases, no lugar onde vai usar).
- [ ] `python -m chubasa escutar`: palavra-chave + uma frase, 10 vezes.
      **Pronto quando:** você é aceito em pelo menos **9 de 10** ("voz: score=... -> aceita").
- [ ] Outra pessoa faz o mesmo 10 vezes.
      **Pronto quando:** **0 de 10** aceitas. Ajuste `SPEAKER_THRESHOLD` entre os scores dos dois.

## 4. Gravação do comando (Bloco 3)

- [ ] `python -m chubasa escutar --salvar-comando comando.wav`: palavra-chave, espere o bip, fale uma frase.
      **Pronto quando:** o `comando.wav` começa e termina junto com a sua frase (sem cortar a 1ª sílaba).
- [ ] Depois do bip, fique calado. **Pronto quando:** volta a dormir em ~5 s.

## 5. Transcrição (Bloco 4)

- [ ] `python -m chubasa transcrever`: fale uma frase de ~5 s.
      **Pronto quando:** o texto sai correto, sem GPU, em até ~3–4 s (no i7-8550U o `small`
      levou ~3,7 s com voz sintética; se passar muito disso, teste `STT_MODEL=base`).

## 6. Conversa (Bloco 5)

- [ ] `python -m chubasa perguntar "Pensa num número de 1 a 10 e me diz." "Multiplica por três." "Qual era o número e qual o resultado?"`
      **Pronto quando:** a 3ª resposta usa o contexto das anteriores. Anote o tempo `[x s]` da
      primeira frase de cada resposta (latência até o primeiro trecho).

## 7. Voz do assistente (Bloco 6)

- [ ] `python -m chubasa falar "Olá, tudo bem? Amanhã deve fazer sol."` — voz clara.
- [ ] `python -m chubasa perguntar --falar "Me conta uma curiosidade sobre o espaço."` — começa a
      falar antes de a resposta terminar de chegar.
      **Pronto quando:** a resposta sai em voz clara.

## 8. Assistente completo (Bloco 7)

- [ ] `python -m chubasa`: palavra-chave → pergunta → resposta falada.
- [ ] Sem fone, deixe ele responder algo longo. **Pronto quando:** ele não se ativa com a própria fala.
- [ ] Modo conversa: logo depois da resposta, faça outra pergunta sem a palavra-chave.
- [ ] "esquece" → ele confirma; a pergunta seguinte não lembra da conversa anterior.
- [ ] Com voz cadastrada: diga a palavra-chave no meio de uma resposta longa → ele para e escuta.
      Diga "para" → ele fica quieto.
- [ ] Ctrl+C encerra limpo ("Até logo!").
- [ ] Deixe rodando **1 hora** (com algumas perguntas no meio) e confira no Gerenciador de Tarefas.
      **Pronto quando:** não trava e a memória do processo fica estável (no teste sintético de 20 min
      ficou em ~950 MB, oscilando ±15 MB).

## 9. Ferramentas (Bloco 8)

- [ ] "Que horas são?" e "Que dia é hoje?" — certos.
- [ ] "Coloca um timer de 1 minuto" — confirma; depois de 1 minuto, bip + aviso falado.
      **Pronto quando:** "coloca um timer de 5 minutos" funciona.
- [ ] "Me lembra de beber água em 2 minutos" — o aviso fala o lembrete.
- [ ] "Abre a calculadora" — abre. "Abre o regedit" — recusa (não está em `TOOLS_PROGRAMS`).

## 10. Segundo plano (Bloco 9)

- [ ] `pythonw -m chubasa --bandeja`: aparece o ícone; a cor muda ao falar com ele; "Pausar
      microfone" deixa o ícone cinza (o indicador de microfone do Windows apaga); "Sair" fecha.
- [ ] `.\scripts\instalar-inicio.ps1 -WhatIf`, depois sem `-WhatIf`.
- [ ] Reinicie o PC, entre no Windows, diga a palavra-chave.
      **Pronto quando:** ele responde sem você abrir nada.
- [ ] Para desfazer: `.\scripts\desinstalar-inicio.ps1`.
