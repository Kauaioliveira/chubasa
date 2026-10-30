# Chubasa

Assistente de voz em Python que fica ouvindo o microfone, acorda com a palavra-chave
"hey chubasa" e responde em voz. A detecção da palavra-chave roda 100% local.

Sistema principal: Windows 10/11. Linux tem suporte (testado no CI); macOS deve funcionar, sem garantia.

## Instalação (Windows, PowerShell)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
python -m chubasa          # imprime "configuração ok"
```

No Linux, instale antes o PortAudio (`sudo apt install libportaudio2`) e, depois do
`pip install -e`, o openWakeWord sem dependências (ele pede `tflite-runtime`, que não existe para
Python 3.12; o Chubasa usa ONNX):

```bash
python -m pip install --no-deps "openwakeword>=0.6,<0.7"
```

## Configuração

Tudo fica no `.env` (veja `.env.example`). Nenhuma chave de API é usada: se a variável
`ANTHROPIC_API_KEY` existir no sistema, o Chubasa avisa no log, porque com ela o uso passa a ser
cobrado por token.

## Comandos

```powershell
python -m chubasa dispositivos   # lista os microfones (use número ou parte do nome em MIC_DEVICE)
python -m chubasa gravar-teste   # grava 5 s em teste.wav
python -m chubasa medidor        # volume do microfone ao vivo (Ctrl+C para sair)
python -m chubasa escutar        # detecta a palavra-chave, grava o comando até você parar de falar
python -m chubasa escutar --salvar-comando comando.wav   # idem, salvando o último comando para ouvir
python -m chubasa cadastrar-voz  # grava 10 a 20 frases suas e salva a impressão de voz
python -m chubasa transcrever    # fale depois do bip e veja o texto (ou passe um WAV de 16 kHz)
python -m chubasa perguntar "pergunta 1" "pergunta 2"   # conversa por texto, mostrando o tempo de cada frase
```

## Palavra-chave

A detecção usa o [openWakeWord](https://github.com/dscripka/openWakeWord) em ONNX, 100% local.
O modelo esperado é `models/hey_chubasa.onnx` (`WAKE_MODEL_PATH`). Enquanto ele não existir, o
Chubasa usa o modelo pronto "hey jarvis" e avisa no log. Para treinar o "hey chubasa" com a sua
voz, siga [treino/LEIA-ME.md](treino/LEIA-ME.md).

Ajustes no `.env`: `WAKE_THRESHOLD` (score mínimo, 0 a 1) e `WAKE_COOLDOWN_S` (tempo ignorando
novas detecções depois de uma). Os modelos baixados ficam em `models/openwakeword/`.

## Gravação do comando

Depois do bip, o Chubasa grava o comando com o [Silero VAD](https://github.com/snakers4/silero-vad)
(ONNX, local, baixado na primeira vez para `models/silero/`): começa quando você fala, guarda um
pedacinho de antes (`VAD_PREROLL_S`) para não cortar a primeira sílaba e para depois de
`VAD_SILENCE_END_S` de silêncio. Se ninguém falar em `VAD_NO_SPEECH_TIMEOUT_S`, volta a dormir;
nenhum comando passa de `VAD_MAX_RECORDING_S`.

## Fala para texto

[faster-whisper](https://github.com/SYSTRAN/faster-whisper) local, em português fixo (`LANGUAGE`).
O modelo é carregado uma vez e baixado na primeira execução para `models/whisper/` (`small`: ~460 MB).
Cada transcrição loga o tempo gasto.

Medido num Intel i7-8550U (4 núcleos, 2017), frase de 4,7 s, CPU `int8`, `beam_size=1`:

| `STT_MODEL` | Tempo | Texto |
| --- | --- | --- |
| `small` (padrão) | 3,6 a 3,8 s | correto |
| `base` | 1,3 s | correto |

Se a resposta parecer lenta, troque para `STT_MODEL=base`. GPU (`STT_DEVICE=cuda`) exige placa
NVIDIA com driver para CUDA 12 e as bibliotecas cuBLAS/cuDNN 12; com driver antigo dá
"CUDA driver version is insufficient", e o padrão `cpu` funciona em qualquer máquina.

## Conversa (Claude pelo plano Pro)

O Chubasa conversa pelo [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python),
usando o **login do Claude Code** da máquina (plano Pro), sem chave de API. Antes do primeiro uso,
faça o login uma vez no terminal:

```powershell
claude        # dentro dele, use /login e entre com a conta do plano
```

- `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN` e `ANTHROPIC_BASE_URL` são removidas do ambiente antes
  de abrir a conversa: o uso nunca vira cobrança por token nem vai para outro servidor.
- Sem ferramentas (não lê arquivos, não roda comandos), sem settings nem CLAUDE.md de usuário/projeto
  e sem servidores MCP: só conversa.
- A conversa fica aberta entre perguntas (histórico e menos espera). A resposta chega em streaming
  e é cortada em frases para a fala começar antes do fim.
- Sem internet, limite do plano, falta de login ou demora viram uma frase falada de aviso.
- Modelo em `BRAIN_MODEL` (padrão `sonnet`; `haiku` responde mais rápido).

## Verificação de voz

Depois de ouvir a palavra-chave, o Chubasa confere se a voz é a da pessoa cadastrada, usando o
trecho da palavra-chave junto com o comando. O modelo é o SpeechBrain ECAPA
(`speechbrain/spkrec-ecapa-voxceleb`), em CPU, baixado na primeira vez (~80 MB) para `models/ecapa/`.

```powershell
python -m pip install -e ".[voz]"   # torch CPU + SpeechBrain
python -m chubasa cadastrar-voz     # leia as frases que aparecerem (padrão: 15)
python -m chubasa escutar           # cada ativação loga "voz: score=... -> aceita/rejeitada"
```

- O cadastro salva só a impressão de voz (`models/voz.npy`, um vetor de números); nenhuma gravação.
- Sem cadastro, qualquer voz ativa o Chubasa (o log avisa). `SPEAKER_VERIFY=false` desliga de vez.
- Calibre `SPEAKER_THRESHOLD` pelos scores do log: a sua voz deve ficar acima, a dos outros abaixo.
- Isso filtra outras pessoas, mas não é senha: uma gravação da sua voz pode passar.
- No Linux, instale o torch CPU antes para não baixar o CUDA:
  `python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu`.

## Testes

```powershell
python -m pytest                 # testes sem hardware (os mesmos do CI)
python -m pytest -m hardware     # usa microfone/alto-falante de verdade
python -m pytest -m modelos      # baixa/carrega modelos reais
```
