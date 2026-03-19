# LAB P1-04: Implementação do O Transformer Completo "From Scratch"

## Como baixar o projeto

```bash
git clone https://github.com/mathcast/LAB-P1-04.git
```

## Objetivo do laboratório

Este laboratório tem como objetivo integrar todos os componentes estudados anteriormente (Self-Attention, Masked Attention, Cross-Attention, Add & Norm e FFN) para construir um **Transformer completo** no modelo **Encoder-Decoder**, executando um fluxo fim-a-fim com geração auto-regressiva.

---

## Conceito geral implementado

```
Entrada (X) → Encoder → Memória (Z)
Entrada parcial (Y) → Decoder → Probabilidades → Próximo token
```

O modelo executa um processo **auto-regressivo**, onde cada novo token é gerado com base nos anteriores.

---

## Fórmulas principais

### Self-Attention

```
Attention(Q, K, V) = softmax((Q K^T) / √d_k) V
```

* **Q (Query)**, **K (Key)** e **V (Value)** são derivados da entrada
* O produto escalar mede similaridade entre tokens
* O softmax transforma em distribuição de atenção

---

### Add & Norm

```
Output = LayerNorm(x + Sublayer(x))
```

- Soma residual preserva informação original
- Normalização estabiliza o treinamento

---

### Feed Forward Network (FFN)

```
FFN(x) = max(0, xW1 + b1)W2 + b2
```

* Expande dimensão (ex: 512 → 2048)
* Aplica não-linearidade (ReLU)
* Retorna à dimensão original

---

## Estrutura do repositório

```
transformer-from-scratch/
├── main.py
├── README.md
├── requirements.txt
│
├── data/
│ └── toy_data.py
│
├── models/
│ ├── attention.py
│ ├── ffn.py
│ ├── add_norm.py
│ ├── encoder.py
│ ├── decoder.py
│ └── transformer.py
│
└── utils/
├── mask.py
└── embeddings.py
```

---

## Descrição dos arquivos

### `models/attention.py`

Implementa o mecanismo de **Scaled Dot-Product Attention**:

* Calcula `Q @ K^T`
* Aplica fator de escala
* Aplica máscara (quando necessário)
* Aplica softmax
* Multiplica por V

---

### `models/ffn.py`

Implementa a **Feed Forward Network**:

* Camada linear → ReLU → camada linear
* Atua em cada posição da sequência independentemente

---

### `models/add_norm.py`

Implementa a operação **Add & Norm**:

* Soma residual entre entrada e saída da subcamada
* Aplica Layer Normalization

---

### `models/encoder.py`

Define o **EncoderBlock**:

```
X → Self-Attention → Add&Norm → FFN → Add&Norm → Z
```

* Processa a entrada de forma bidirecional
* Pode ser empilhado (multi-layer encoder)

---

### `models/decoder.py`

Define o **DecoderBlock**:

```
Y → Masked Attention → Add&Norm → Cross Attention → Add&Norm → FFN → Add&Norm
```

* Masked Attention impede acesso ao futuro
* Cross Attention conecta com a saída do encoder (Z)

---

### `models/transformer.py`

Integra o modelo completo:

```
Encoder → Decoder
```

* Recebe entrada (src) e saída parcial (tgt)
* Retorna logits (probabilidades antes do softmax)

---

### `utils/mask.py`

Gera máscara causal:

```
[1, 0, 0]
[1, 1, 0]
[1, 1, 1]
```

* Impede que o modelo veja tokens futuros

---

### `utils/embeddings.py`

Arquivo vazio intencionalmente:

```
Este arquivo está vazio intencionalmente.
Neste projeto, embeddings reais não estão sendo utilizados.
Em vez disso, os dados de entrada já são tensores aleatórios com dimensão 512:
encoder_input = torch.rand(1, 5, 512)
decoder_input = torch.rand(1, 1, 512)
Isso significa que os vetores já estão no formato esperado pelo modelo,
simulando embeddings sem precisar de uma camada de embedding explícita.
```

---

### `data/toy_data.py`

Gera dados simulados:

* Entrada do encoder (frase fictícia)
* Entrada inicial do decoder (`<START>`)

---

### `main.py`

Executa o modelo completo:

* Cria o Transformer
* Executa encoder + decoder
* Implementa geração auto-regressiva

---

## Como rodar

### 1. Criar ambiente virtual

**Windows:**
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Linux/macOS:

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Executar o projeto
```
python main.py
```

## Funcionamento da inferência

### Loop auto-regressivo

1. Decoder inicia com <START>
2. Modelo prevê próximo token
3. Token é concatenado na entrada
4. Processo repete até atingir tamanho máximo

### Fluxo completo

#### Encoder

```
Entrada (X) → Self-Attention → FFN → Z
```

#### Decoder

```
Entrada parcial (Y)
→ Masked Attention
→ Cross Attention (com Z)
→ FFN
→ Linear + Softmax
```

## Observações importantes

* O modelo não foi treinado
* Os dados são aleatórios
* As saídas são aleatórias, mas estruturalmente corretas

## Validações realizadas

* Shapes dos tensores mantidos corretamente
* Máscara causal funcionando corretamente
* Atenção normalizada (soma ≈ 1)
* Sequência crescendo a cada passo
* Fluxo Encoder → Decoder correto

## Requisitos técnicos

* Linguagem: Python 3
* Bibliotecas: PyTorch, NumPy
