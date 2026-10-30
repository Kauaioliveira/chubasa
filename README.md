# Chubasa

Assistente de voz em Python que fica ouvindo o microfone, acorda com a palavra-chave
**"hey chubasa"**, confere se é a sua voz, entende o que você pediu e responde falando.

- Palavra-chave, detecção de fala, verificação de voz, transcrição e voz rodam **no computador**.
- Só o texto do que você pediu, depois da palavra-chave, sai da máquina: o "cérebro" é o Claude,
  usado pelo **login do plano Pro** (sem chave de API, sem cobrança por token).
- Sistema principal: **Windows 10/11**. Linux tem suporte (testado no CI). macOS deve funcionar, sem garantia.

## Como funciona

```
dormindo ──"hey chubasa"──> ouvindo ──fala + 1 s de silêncio──> pensando ──streaming──> falando
    ^                          │ (ninguém falou em 5 s,             │                     │
    │                          │  voz de outra pessoa)              │                     │
    └──────────────────────────┴────────────<── conversando (6 s sem palavra-chave) <─────┘
```

Enquanto fala, o assistente não escuta a palavra-chave (para não se ativar com a própria voz).
Com a voz cadastrada, dá para interromper dizendo a palavra-chave (score alto + voz conferida).

## Instalação (Windows, PowerShell)

Pré-requisitos: Python 3.12 (python.org, marque "Add to PATH") e o Claude Code instalado.

```powershell
cd C:\Projects\chubasa
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[voz]"      # tudo, incluindo a verificação de voz (torch CPU, ~200 MB)
Copy-Item .env.example .env
python -m chubasa config               # deve imprimir "configuração ok"
```

**Login (uma vez):** o assistente usa a conta do Claude Code desta máquina.

```powershell
claude        # dentro dele, digite /login e entre com a conta do plano Pro; depois /exit
```

Sem login, o assistente funciona e avisa em voz: "Não estou conectado à conta...".

Na primeira execução são baixados os modelos (para `models/`, fora do git): palavra-chave (~9 MB),
detector de fala (~2 MB), transcrição `small` (~460 MB), voz (~60 MB) e verificação de voz (~85 MB).

### Linux

```bash
sudo apt install libportaudio2
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu   # evita baixar CUDA
python -m pip install -e ".[voz]"
python -m pip install --no-deps "openwakeword>=0.6,<0.7"   # ele pede tflite-runtime, que não existe para 3.12
cp .env.example .env
```

## Uso

```powershell
python -m chubasa                      # roda o assistente no terminal (Ctrl+C para sair)
pythonw -m chubasa --bandeja           # roda sem janela, com ícone na bandeja (ou: chubasa-bandeja)
```

Diga **"hey chubasa"** (ou "hey jarvis" enquanto o modelo próprio não estiver treinado), espere o bip
e faça o pedido. Depois da resposta, você tem alguns segundos para continuar a conversa sem a
palavra-chave. Comandos de voz:

- **"esquece"** / "nova conversa": apaga o histórico da conversa.
- **"para"** / "cancela": para de falar.
- "que horas são?", "coloca um timer de 5 minutos", "me lembra de tirar o bolo em 20 minutos",
  "abre a calculadora".

### Ícone na bandeja

Cor do ícone: azul = esperando a palavra-chave, verde = ouvindo, laranja = pensando,
roxo = falando, cinza riscado = microfone pausado. Menu: **Pausar/Retomar microfone** (com o
microfone pausado o dispositivo é fechado de verdade) e **Sair**.

### Iniciar com o Windows

```powershell
.\scripts\instalar-inicio.ps1 -WhatIf     # mostra o que seria feito
.\scripts\instalar-inicio.ps1             # cria a tarefa "Chubasa" no Agendador (ao fazer logon)
.\scripts\desinstalar-inicio.ps1          # remove
```

A tarefa roda `.venv\Scripts\pythonw.exe -m chubasa --bandeja` como o seu usuário (sem admin).

### Linux (systemd do usuário)

Veja as instruções no topo de [scripts/chubasa.service](scripts/chubasa.service)
(`systemctl --user enable --now chubasa`). No Linux roda sem bandeja.

### Logs

O assistente grava log com rotação (1 MB x 5 arquivos) em:

- Windows: `%LOCALAPPDATA%\Chubasa\logs\chubasa.log`
- Linux: `~/.local/state/chubasa/chubasa.log` (ou `$XDG_STATE_HOME/chubasa`)

Cada interação registra os scores da palavra-chave e da voz, o tempo de transcrição e a latência
até o primeiro trecho da resposta: é por eles que se calibram os limiares.

## Comandos de teste e diagnóstico

```powershell
python -m chubasa config                 # confere o .env
python -m chubasa dispositivos           # lista os microfones (use número ou parte do nome em MIC_DEVICE)
python -m chubasa gravar-teste           # grava 5 s em teste.wav
python -m chubasa medidor                # volume do microfone ao vivo
python -m chubasa escutar                # só a palavra-chave + gravação do comando + verificação de voz
python -m chubasa escutar --salvar-comando comando.wav   # idem, salvando o último comando para ouvir
python -m chubasa cadastrar-voz          # grava 10 a 20 frases suas e salva a impressão de voz
python -m chubasa transcrever            # fale depois do bip e veja o texto (ou passe um WAV de 16 kHz)
python -m chubasa perguntar "pergunta 1" "pergunta 2"    # conversa por texto, com o tempo de cada frase
python -m chubasa perguntar --falar "pergunta"           # idem, falando a resposta
python -m chubasa falar "Olá!"           # testa a voz
```

Gravações só são salvas quando você pede (`gravar-teste`, `--salvar-comando`, kit de treino) e
nunca vão para o git.

## Configuração (`.env`)

Copie de `.env.example`; cada variável está comentada lá. Caminhos relativos são relativos à
pasta do projeto.

| Variável | Padrão | Para que serve |
| --- | --- | --- |
| `WAKE_WORD`, `LANGUAGE` | `hey chubasa`, `pt` | palavra-chave (texto exibido) e idioma (obrigatórios) |
| `MIC_DEVICE` | vazio (padrão do sistema) | microfone: número ou parte do nome |
| `LOG_LEVEL` | `INFO` | `DEBUG` mostra o score da palavra-chave a cada segundo |
| `MODELS_DIR` | `models` | onde ficam os modelos baixados/treinados |
| `WAKE_MODEL_PATH` | `models/hey_chubasa.onnx` | modelo da palavra-chave (sem ele, usa "hey jarvis") |
| `WAKE_THRESHOLD` | `0.5` | score mínimo da palavra-chave; suba se disparar à toa |
| `WAKE_COOLDOWN_S` | `2.0` | tempo ignorando novas detecções |
| `SPEAKER_VERIFY` | `true` | conferir a voz (só vale com voz cadastrada) |
| `SPEAKER_PROFILE_PATH` | `models/voz.npy` | impressão de voz |
| `SPEAKER_THRESHOLD` | `0.40` | similaridade mínima com a sua voz |
| `VAD_THRESHOLD` | `0.5` | sensibilidade do detector de fala |
| `VAD_SILENCE_END_S` | `1.0` | silêncio que encerra o comando |
| `VAD_NO_SPEECH_TIMEOUT_S` | `5.0` | sem fala depois do bip: volta a dormir |
| `VAD_MAX_RECORDING_S` | `30` | duração máxima de um comando |
| `VAD_PREROLL_S` | `0.3` | áudio antes da fala que entra na gravação |
| `STT_MODEL` | `small` | `base` é ~3x mais rápido em máquina fraca |
| `STT_DEVICE`, `STT_COMPUTE_TYPE` | `cpu`, `int8` | `cuda`/`float16` só com NVIDIA e CUDA 12 |
| `STT_BEAM_SIZE`, `STT_CPU_THREADS` | `1`, `0` | precisão x velocidade; threads (0 = automático) |
| `BRAIN_MODEL` | `sonnet` | `haiku` responde mais rápido; `opus` mais elaborado |
| `BRAIN_TIMEOUT_FIRST_S`, `BRAIN_TIMEOUT_TOTAL_S` | `30`, `90` | desiste e avisa depois disso |
| `LOOP_CONVERSATION_S` | `6` | modo conversa depois de responder (0 = desliga) |
| `LOOP_BARGE_IN`, `LOOP_BARGE_IN_THRESHOLD` | `true`, `0.8` | interromper a fala com a palavra-chave |
| `TOOLS_ENABLED` | `true` | hora/data, timers/lembretes, abrir programas |
| `TOOLS_PROGRAMS` | bloco de notas, calculadora, chrome | programas permitidos, `nome:comando;...` |
| `TTS_ENGINE` | `piper` | `edge` = voz pela internet (`pip install -e ".[edge]"`) |
| `TTS_VOICE`, `TTS_EDGE_VOICE` | `pt_BR-faber-medium`, `pt-BR-AntonioNeural` | vozes |

## Palavra-chave própria ("hey chubasa")

O modelo pronto "hey jarvis" serve para testar. Para treinar o "hey chubasa" com a sua voz
(gravação no PC + treino grátis no Google Colab), siga [treino/LEIA-ME.md](treino/LEIA-ME.md).
O resultado vai em `models/hey_chubasa.onnx`.

## Verificação de voz

Com a voz cadastrada (`python -m chubasa cadastrar-voz`), só você ativa o assistente: a palavra-chave
e o comando são comparados com a sua impressão de voz (SpeechBrain ECAPA, local). Sem cadastro,
qualquer voz ativa (o log avisa). Filtra outras pessoas, mas não é senha: uma gravação da sua voz
pode passar.

## Cérebro, ferramentas e privacidade

- A conversa usa o Claude pelo Claude Agent SDK com o login do plano Pro. `ANTHROPIC_API_KEY`,
  `ANTHROPIC_AUTH_TOKEN` e `ANTHROPIC_BASE_URL` são removidas do ambiente antes de abrir a conversa.
- As ferramentas embutidas (arquivos, terminal, web) ficam desligadas; não são carregadas
  configurações nem servidores MCP da máquina. Só estas ferramentas do próprio assistente ficam
  liberadas: **hora/data**, **timers e lembretes** (avisam em voz e bip), **abrir programa** (só os
  de `TOOLS_PROGRAMS`). Nada apaga arquivo nem envia mensagem.
- A conversa fica aberta entre perguntas (tem histórico); "esquece" começa outra.
- Sem internet, limite do plano, falta de login ou demora viram uma frase falada de aviso.
- O uso conta nos mesmos limites do plano Pro que o chat e o Claude Code.

## Desempenho medido

Notebook com Intel i7-8550U (4 núcleos, 2017), 16 GB de RAM, sem GPU utilizável:

| Etapa | Medida |
| --- | --- |
| Palavra-chave | ~3 ms por frame de 80 ms |
| Transcrição de frase de 4,7 s, `small` int8 | 3,6 a 3,8 s (texto correto) |
| Idem, `base` int8 | 1,3 s (texto correto) |
| Verificação de voz | ~0,4 s |
| Voz (Piper), depois de aquecida | 4,5 s de fala sintetizados em 0,35 s |
| Do primeiro trecho da resposta até começar a falar | ~0,4 s |
| Carregar todos os modelos | ~26 s (`small` + verificação de voz + voz) |
| Memória do processo | ~950 MB e estável (teste de 20 min, 52 interações, `base` + verificação de voz) |
| Transcrição dentro do loop, `base` | mediana 0,97 s por comando |

A latência do cérebro (pergunta até o primeiro trecho da resposta) depende do plano e da rede e é
registrada no log a cada pergunta.

## Limitações conhecidas

- **GPU:** a transcrição em GPU exige NVIDIA com driver para CUDA 12. Uma GeForce MX150 com driver
  511.69 (CUDA 11.6) falha com "CUDA driver version is insufficient"; atualizar o driver resolveria
  em tese, mas o padrão `cpu` funciona em qualquer máquina.
- **Transcrição lenta em CPU antiga:** `small` leva ~3,7 s numa frase de 5 s no i7-8550U. Use
  `STT_MODEL=base` se a espera incomodar.
- **Eco:** sem fone, o microfone capta a própria resposta. A palavra-chave fica muda enquanto o
  assistente fala; a interrupção por voz só é liberada com a voz cadastrada.
- **Interromper só com voz cadastrada:** sem cadastro não há como distinguir você da própria voz
  do assistente, então a interrupção fica desligada.
- **Memória:** o assistente usa ~1 GB. Com pouca RAM livre a transcrição pode falhar ("failed to
  allocate memory"); ele avisa em voz, registra no log e continua. Feche programas ou use `STT_MODEL=base`.
- **"hey jarvis" até treinar:** o modelo próprio depende do kit de treino.

## Testes

```powershell
python -m pip install -e ".[dev]"
python -m pytest                 # testes sem hardware (os mesmos do CI)
python -m pytest -m hardware     # usa microfone/alto-falante de verdade
python -m pytest -m modelos      # baixa/carrega modelos reais
python -m pytest -m interface    # abre o ícone da bandeja de verdade
```

O que validar com voz real está em [CHECKLIST-VALIDACAO.md](CHECKLIST-VALIDACAO.md).
