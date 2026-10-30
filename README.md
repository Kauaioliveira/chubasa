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

No Linux, instale antes o PortAudio (`sudo apt install libportaudio2`).

## Configuração

Tudo fica no `.env` (veja `.env.example`). Nenhuma chave de API é usada: se a variável
`ANTHROPIC_API_KEY` existir no sistema, o Chubasa avisa no log, porque com ela o uso passa a ser
cobrado por token.

## Comandos

```powershell
python -m chubasa dispositivos   # lista os microfones (use número ou parte do nome em MIC_DEVICE)
python -m chubasa gravar-teste   # grava 5 s em teste.wav
python -m chubasa medidor        # volume do microfone ao vivo (Ctrl+C para sair)
python -m chubasa escutar        # detecta a palavra-chave e mostra o score (bip ao detectar)
```

## Palavra-chave

A detecção usa o [openWakeWord](https://github.com/dscripka/openWakeWord) em ONNX, 100% local.
O modelo esperado é `models/hey_chubasa.onnx` (`WAKE_MODEL_PATH`). Enquanto ele não existir, o
Chubasa usa o modelo pronto "hey jarvis" e avisa no log. Para treinar o "hey chubasa" com a sua
voz, siga [treino/LEIA-ME.md](treino/LEIA-ME.md).

Ajustes no `.env`: `WAKE_THRESHOLD` (score mínimo, 0 a 1) e `WAKE_COOLDOWN_S` (tempo ignorando
novas detecções depois de uma). Os modelos baixados ficam em `models/openwakeword/`.

## Testes

```powershell
python -m pytest                 # testes sem hardware (os mesmos do CI)
python -m pytest -m hardware     # usa microfone/alto-falante de verdade
python -m pytest -m modelos      # baixa/carrega modelos reais
```
