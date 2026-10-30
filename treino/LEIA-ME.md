# Treinar a palavra-chave "hey chubasa"

Enquanto `models/hey_chubasa.onnx` não existir, o Chubasa usa o modelo pronto **"hey jarvis"**
(o log avisa). Este kit gera o modelo próprio em três passos: gravar a sua voz no PC, treinar no
Google Colab (GPU grátis) e copiar o arquivo de volta.

O treino segue o fluxo oficial do openWakeWord: milhares de vozes sintéticas dizendo a frase,
misturadas com ruído e eco de sala, contra milhares de horas de fala que não é a palavra-chave.
Suas gravações entram junto e deixam o modelo mais afinado à sua voz e ao seu microfone.

## 1. Gravar a sua voz (no PC, ~10 minutos)

Na pasta do projeto, com o venv ativo:

```powershell
.venv\Scripts\Activate.ps1
python treino\gravar_amostras.py --positivas 30 --negativas 30
```

- Para cada amostra: aperte Enter e fale. Cada gravação dura 2,5 s (`--segundos` muda isso).
- **Positivas:** diga "hey chubasa" do jeito que você vai falar no dia a dia. Varie um pouco:
  mais perto e mais longe do microfone, mais rápido e mais devagar, voz normal e mais baixa.
- **Negativas:** leia a frase que aparece na tela (frases parecidas, como "hey chuva", e fala comum).
- Grave no lugar e com o microfone que vai usar com o Chubasa.
- Amostra muda ou baixa demais é descartada e repetida automaticamente.
- Rodar de novo **acrescenta** amostras (a numeração continua). Ctrl+C interrompe sem perder o que já foi gravado.
- Para outro microfone: `--dispositivo "nome ou número"` (lista: `python -m chubasa dispositivos`).

Resultado: `treino\amostras\positivas\*.wav`, `treino\amostras\negativas\*.wav` e o pacote
`treino\amostras.zip`. **Nada disso vai para o git** (está no `.gitignore`): são gravações da sua voz.

## 2. Treinar no Colab (1 a 3 horas, quase tudo sem você)

1. Abra https://colab.research.google.com > Arquivo > Fazer upload de notebook > escolha
   `treino\treinar_hey_chubasa.ipynb`.
2. Ambiente de execução > Alterar o tipo de ambiente de execução > **GPU T4** > Salvar.
3. Rode as células em ordem (Shift+Enter):
   - **1. Parâmetros:** os padrões servem. Para um modelo mais robusto, aumente `N_AMOSTRAS`
     (ex.: 20000) e `PASSOS` (ex.: 50000); demora mais.
   - **2. Ambiente:** cria um Python 3.10 separado com as bibliotecas de treino (~5 min).
   - **3. Dados:** baixa ecos de sala, ruído de fundo e 17 GB de features de fala (~15 min).
   - **4. Suas gravações:** clique em "Escolher arquivos" e envie o `amostras.zip`.
   - **6. Vozes sintéticas:** depois de gerar, ouça os exemplos. Se "chubasa" estiver soando
     errado, ajuste `FRASES`/`PRONUNCIAS` na etapa 1, apague `/content/modelo` e gere de novo.
   - **8 e 9:** aumento de dados e treino (a parte mais demorada).
   - **10:** mostra o score do modelo nas suas gravações e baixa o `hey_chubasa.onnx`.

Se uma célula falhar, corrija e rode de novo a partir dela: o que já foi baixado ou gerado é
reaproveitado. Se o Colab reiniciar o ambiente (desconexão longa), tudo em `/content` se perde
e é preciso começar da etapa 1.

## 3. Usar o modelo

1. Copie o `hey_chubasa.onnx` baixado para `models\hey_chubasa.onnx` na pasta do projeto
   (ou aponte `WAKE_MODEL_PATH` no `.env` para onde ele estiver).
2. Rode `python -m chubasa escutar`. O aviso do "hey jarvis" some e o log mostra
   `Wake word: "hey chubasa"`.
3. Diga "hey chubasa" 10 vezes: o plano pede pelo menos 9 detecções. Depois deixe 10 minutos
   de conversa ou TV: no máximo 1 falso disparo.
4. Calibre `WAKE_THRESHOLD` no `.env` olhando os scores do log:
   - não detecta ou detecta pouco → **desça** (ex.: 0.4, 0.3);
   - dispara à toa → **suba** (ex.: 0.6, 0.7);
   - com `LOG_LEVEL=DEBUG` aparece o score máximo de cada segundo.

Se não chegar a 9 de 10 mesmo com limiar baixo, grave mais positivas (50 a 100) e treine de novo
com `N_AMOSTRAS` e `PASSOS` maiores.

## Por que o notebook é diferente do oficial

O notebook oficial (`automatic_model_training.ipynb` do openWakeWord) não roda mais como está:

| Problema no oficial | O que este notebook faz |
| --- | --- |
| Dependências de treino não instalam no Python 3.12 do Colab | Cria um Python 3.10 isolado com `uv` e versões fixadas (resolução conferida para Linux/3.10) |
| O modelo do DeepPhonemizer, usado para palavras fora do dicionário, dá HTTP 403 | Injeta a pronúncia de "chubasa" no dicionário CMU antes do treino |
| O `bal_train09.tar` do AudioSet não existe mais (404) | Lê os arquivos parquet novos do AudioSet e converte para WAV 16 kHz |
| FMA via script de dataset (formato descontinuado) | Usa só o AudioSet como ruído de fundo |
| Conversão final para tflite exige TensorFlow | Pulada: o Chubasa usa ONNX |

Em português, "Chubasa" soa "shu-BA-za". O TTS do treino é em inglês, por isso as frases-alvo
incluem a grafia fonética "hey shoo bah zah", além de "hey chubasa".
