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

## Testes

```powershell
python -m pytest                 # testes sem hardware (os mesmos do CI)
python -m pytest -m hardware     # usa microfone/alto-falante de verdade
python -m pytest -m modelos      # baixa/carrega modelos reais
```
