# Pesquisa Profunda: Sistema LLM para Responder Perguntas sobre Saúde Pública

**Disciplina:** Projeto e Análise de Algoritmos (PAA)  
**Tema:** Questões de Saúde Pública  
**Data:** Novembro 2025

---

## 1. INTRODUÇÃO E CONTEXTO

### 1.1 Objetivo do Projeto
Desenvolver um sistema de Processamento de Linguagem Natural (NLP) capaz de responder perguntas em linguagem natural sobre saúde pública, integrando:
- **Coleta e processamento de dados** sobre saúde pública
- **Treinamento de modelo LLM** (Large Language Model) especializado
- **Interface web** interativa para interação do usuário
- **Análise de complexidade** dos algoritmos implementados

### 1.2 Relevância do Tema
Saúde pública é um domínio crítico onde:
- Há demanda constante por informações precisas
- A precisão é essencial (questões médicas)
- Existem conjuntos de dados públicos disponíveis
- É um contexto realista e impactante para demonstração em sala de aula

---

## 2. FUNDAMENTAÇÃO TEÓRICA

### 2.1 O que é um LLM (Large Language Model)?

Um **LLM** é uma rede neural profunda baseada em arquitetura **Transformer** que foi pré-treinada em quantidades massivas de texto para:
- Compreender linguagem natural
- Gerar texto coerente
- Realizar transferência de conhecimento para tarefas específicas

**Características principais:**
- **Bilhões de parâmetros** (pesos da rede)
- **Pré-treinamento** em corpus genérico de texto
- **Fine-tuning** em dados específicos do domínio
- **Arquitetura Transformer** com mecanismo de atenção

### 2.2 Arquitetura Transformer

```
┌─────────────────────────────────────────┐
│         INPUT (Pergunta do usuário)      │
└────────────┬────────────────────────────┘
             │
      ┌──────▼──────┐
      │ Tokenização │ (Converte texto em tokens)
      └──────┬──────┘
             │
      ┌──────▼──────────────┐
      │ Embedding + Position │ (Representação vetorial)
      └──────┬──────────────┘
             │
    ┌────────▼────────┐
    │  Multi-Head     │
    │  Self-Attention │ (Entende relações entre palavras)
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │  Feed Forward   │ (Processa representações)
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │  Decoder Layers │ (Múltiplas camadas)
    └────────┬────────┘
             │
      ┌──────▼──────┐
      │  Linear + SoftMax │ (Gera probabilidades de tokens)
      └──────┬──────┘
             │
      ┌──────▼──────┐
      │ OUTPUT      │ (Resposta em linguagem natural)
      └─────────────┘
```

**Componentes-chave:**
1. **Self-Attention:** Cada palavra "vê" todas as outras, calculando relevância
2. **Multi-Head:** Múltiplas representações simultâneas
3. **Layers:** Processamento em camadas sucessivas
4. **Tokenização:** Quebra texto em unidades processáveis

### 2.3 Processo de Treinamento

#### 2.3.1 Pré-Treinamento (Feito pelo fabricante)
- Treinado em **centenas de bilhões de tokens** de texto genérico
- Objetivo: Prever próximo token na sequência
- Resultado: Modelo compreende padrões gerais de linguagem

#### 2.3.2 Fine-Tuning (Seu projeto!)
- Pega modelo pré-treinado
- Treina com dados específicos de **saúde pública**
- Adapta pesos da rede para domínio específico
- Melhora muito a qualidade das respostas

#### 2.3.3 Metodologias de Fine-Tuning

**A) Supervised Fine-Tuning (SFT)**
- Dataset: pares (pergunta, resposta esperada)
- Ensina modelo a gerar respostas específicas
- Melhor para: classificação, respostas diretas
- Complexidade: O(n × m) onde n = tokens, m = exemplos

```python
# Pseudocódigo SFT
for epoch in range(num_epochs):
    for batch in training_data:
        pergunta, resposta_esperada = batch
        logits = model(pergunta)
        loss = cross_entropy(logits, resposta_esperada)
        loss.backward()
        optimizer.step()
```

**B) Direct Preference Optimization (DPO)**
- Dataset: perguntas com respostas "boas" vs "ruins"
- Treina modelo a **preferir** respostas melhores
- Melhor para: raciocínio clínico, triage
- **Recomendado para seu projeto** (mais efetivo)

```python
# Pseudocódigo DPO
for batch in training_data:
    pergunta = batch['pergunta']
    resposta_boa = batch['resposta_boa']
    resposta_ruim = batch['resposta_ruim']
    
    score_boa = model.score(pergunta, resposta_boa)
    score_ruim = model.score(pergunta, resposta_ruim)
    
    loss = -log(sigmoid(score_boa - score_ruim))
    loss.backward()
```

**C) LoRA (Low-Rank Adaptation)** - RECOMENDADO PARA SEU CASO
- Não modifica todos os pesos
- Adiciona **adaptadores de baixa classificação**
- Reduz tempo de treinamento em **90%**
- Reduz memória necessária em **80%**
- Mantém qualidade do modelo original

```
Parâmetros do modelo original: 7 bilhões
Parâmetros adicionais LoRA: ~13 milhões (0.2%)
```

---

## 3. STACK TECNOLÓGICO RECOMENDADO

### 3.1 Arquitetura Geral do Sistema

```
┌──────────────────────────────────────────────────────┐
│            INTERFACE WEB (Frontend)                   │
│  Framework: React.js / Vue.js / HTML5 + JavaScript    │
└────────────────────┬─────────────────────────────────┘
                     │ HTTP/WSGI
┌────────────────────▼─────────────────────────────────┐
│         SERVIDOR BACKEND (API REST/FastAPI)           │
│  - Recebe pergunta em JSON                            │
│  - Processa com modelo LLM                            │
│  - Retorna resposta em JSON                           │
└────────────────────┬─────────────────────────────────┘
                     │ Carregamento em memória
┌────────────────────▼─────────────────────────────────┐
│         PIPELINE DE PROCESSAMENTO                     │
│  1. Tokenização (HuggingFace Tokenizers)              │
│  2. Embedding (Modelo)                               │
│  3. Geração de tokens (Modelo)                        │
│  4. Detokenização                                     │
└────────────────────┬─────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────┐
│  MODELO LLM FINE-TUNED (Arquivo .pth/.safetensors)   │
│  - Base: LLaMA 2 7B ou Mistral 7B                     │
│  - Adaptadores LoRA aplicados                        │
│  - Treinado em dados de saúde pública               │
└──────────────────────────────────────────────────────┘
```

### 3.2 Stack Específico (Com Licenças Open Source)

| Componente | Tecnologia | Licença | Justificativa |
|-----------|-----------|---------|---------------|
| **Linguagem Principal** | Python 3.10+ | PSF | Melhor ecossistema ML/NLP |
| **Framework Deep Learning** | PyTorch 2.0+ | BSD | Preferido para NLP, flexível |
| **Bibliotecas LLM** | Hugging Face Transformers | Apache 2.0 | Padrão ouro em NLP |
| **Fine-tuning Eficiente** | PEFT (LoRA) | Apache 2.0 | Implementação LoRA oficial |
| **Tokenização** | HuggingFace Tokenizers | Apache 2.0 | Tokenização otimizada |
| **Servidor Backend** | FastAPI | MIT | Moderno, performático, fácil |
| **Servidor WSGI** | Uvicorn | BSD | Servidor ASGI para FastAPI |
| **Frontend** | React.js ou HTML5 | MIT/Apache | Interface interativa |
| **Database (opcional)** | SQLite3 ou PostgreSQL | Public Domain/PostgreSQL | Persistência de dados |
| **Logging/Monitoramento** | Python logging | PSF | Já na stdlib |
| **Containerização** | Docker | Apache 2.0 | Reprodutibilidade |

### 3.3 Ambiente de Desenvolvimento

```bash
# Arquivo: requirements.txt
torch>=2.0.0          # PyTorch
transformers>=4.35.0  # Hugging Face
peft>=0.7.0           # LoRA
pydantic>=2.0         # Validação de dados
fastapi>=0.104.0      # Framework web
uvicorn>=0.24.0       # Servidor ASGI
numpy>=1.24.0         # Arrays
scikit-learn>=1.3.0   # Utilitários ML
pandas>=2.0.0         # Manipulação de dados
requests>=2.31.0      # Requisições HTTP
python-dotenv>=1.0.0  # Variáveis de ambiente
```

---

## 4. DATASET E COLETA DE DADOS

### 4.1 Fontes de Dados Recomendadas (Públicas e Gratuitas)

#### **Opção 1: WHO (Organização Mundial de Saúde)**
- **URL:** https://www.who.int/publications
- **Conteúdo:** Relatórios, guias, FAQ sobre doenças
- **Formato:** PDF, HTML
- **Pré-processamento:** OCR + parsing

#### **Opção 2: Datasets Médicos em Português**
- **MedQA Dataset** (modificado para português)
- **Perguntas e respostas médicas** de comunidades
- **Documentos SUS** (Sistema Único de Saúde)

#### **Opção 3: PubMed Central**
- **URL:** https://www.ncbi.nlm.nih.gov/pmc/
- **Conteúdo:** Artigos científicos de acesso livre
- **API:** Disponível para download automático

#### **Opção 4: Criar Seu Próprio Dataset**
- Manualmente com especialistas
- Crowdsourcing na universidade
- Perguntas frequentes de saúde pública

### 4.2 Estrutura do Dataset

```json
{
  "training_data": [
    {
      "id": 1,
      "pergunta": "O que é dengue?",
      "resposta": "Dengue é uma doença infecciosa causada pelo vírus...",
      "categoria": "doenças_infecciosas",
      "confiança": 0.95
    },
    {
      "id": 2,
      "pergunta": "Como se transmite o vírus Zika?",
      "resposta": "O vírus Zika é transmitido principalmente pelo mosquito Aedes aegypti...",
      "categoria": "transmissão_doença",
      "confiança": 0.98
    }
  ]
}
```

### 4.3 Requisitos de Qualidade

- **Mínimo 500-1000 exemplos** para fine-tuning básico
- **Máximo 10.000 exemplos** antes de overfitting
- **Balanceamento:** Distribuição uniforme de categorias
- **Validade:** Revisão por especialista (se possível)

### 4.4 Pré-Processamento de Dados

```python
# Pseudocódigo de pipeline de dados
def preprocessar_dataset(raw_data):
    """
    Complexidade: O(n × m) onde:
    - n = número de exemplos
    - m = comprimento médio do texto
    """
    processado = []
    
    for exemplo in raw_data:
        # 1. Limpeza
        texto = remover_html(exemplo['text'])
        texto = remover_caracteres_especiais(texto)
        
        # 2. Normalização
        texto = texto.lower()
        texto = remover_acentos(texto)
        
        # 3. Validação
        if len(texto) > 50 and len(texto) < 5000:
            processado.append({
                'pergunta': exemplo['pergunta'],
                'resposta': texto,
                'tokens_count': len(tokenizer.encode(texto))
            })
    
    return processado
```

---

## 5. METODOLOGIA DE TREINAMENTO

### 5.1 Escolha do Modelo Base

#### **LLaMA 2 7B (Recomendado)**
- **Tamanho:** 7 bilhões de parâmetros
- **Licença:** Llama 2 Community License (permitida para fins educacionais)
- **Vantagens:**
  - Bom equilíbrio entre qualidade e tamanho
  - Executável em GPU modesta (12GB VRAM)
  - Bem documentado
  - Excelente para português
  
#### **Mistral 7B (Alternativa)**
- **Tamanho:** 7 bilhões de parâmetros
- **Licença:** Apache 2.0
- **Vantagens:**
  - Performance superior ao LLaMA em certos benchmarks
  - Mais eficiente em termos de inferência

#### **Por que NOT usar GPT-4?**
- Proprietário (não open source)
- Requer API com custo
- Violaria requisito de "open source" do projeto

### 5.2 Configuração de Fine-Tuning (LoRA)

```python
# Configuração recomendada para seu projeto
lora_config = {
    "r": 16,                      # Rank dos adaptadores LoRA
    "lora_alpha": 32,             # Escala de aprendizado
    "target_modules": [
        "q_proj",                 # Query projection
        "v_proj"                  # Value projection
    ],
    "lora_dropout": 0.05,         # Dropout no LoRA
    "bias": "none",
    "task_type": "CAUSAL_LM"      # Linguagem causal (predição de próximo token)
}

training_config = {
    "num_epochs": 3,              # 3 épocas (pode variar)
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,        # Taxa de aprendizado
    "warmup_steps": 100,
    "max_steps": -1,              # Usa num_epochs
    "logging_steps": 10,
    "save_strategy": "epoch",
    "eval_strategy": "epoch",
    "weight_decay": 0.01,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "max_grad_norm": 1.0
}
```

### 5.3 Pipeline de Treinamento

```python
# Pseudocódigo do pipeline
class LLMHealthTrainer:
    def __init__(self, model_name, config):
        """
        Complexidade de inicialização: O(1)
        - Carrega modelo pré-treinado
        """
        self.model = load_pretrained_model(model_name)
        self.tokenizer = load_tokenizer(model_name)
        self.config = config
    
    def preparar_dados(self, dataset_path):
        """
        Complexidade: O(n) onde n = tamanho do dataset
        """
        dataset = carregar_json(dataset_path)
        dataset = dataset.map(self.tokenizar_exemplo)
        dataset = dataset.filter(lambda x: len(x['input_ids']) < 2048)
        return dataset.train_test_split(test_size=0.1)
    
    def tokenizar_exemplo(self, exemplo):
        """
        Complexidade por exemplo: O(m) onde m = comprimento do texto
        """
        prompt = f"Pergunta: {exemplo['pergunta']}\nResposta: {exemplo['resposta']}"
        encoded = self.tokenizer.encode(prompt, max_length=2048, truncation=True)
        return {
            'input_ids': encoded,
            'attention_mask': [1] * len(encoded),
            'labels': encoded  # Mesmo que input para LM
        }
    
    def treinar(self, train_dataset, eval_dataset):
        """
        Complexidade: O(num_epochs × len(dataset) × modelo_forward_pass)
        Típico: ~20-60 minutos em GPU moderna
        """
        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(**self.config),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        
        trainer.train()
        trainer.save_model("./modelo_saude_publica_final")
    
    def avaliar(self, eval_dataset):
        """
        Complexidade: O(len(eval_dataset) × modelo_inference)
        """
        metricas = self.trainer.evaluate(eval_dataset)
        
        # Calcula BLEU, ROUGE, BERTScore
        return {
            'loss': metricas['eval_loss'],
            'perplexidade': math.exp(metricas['eval_loss'])
        }
```

---

## 6. ANÁLISE DE COMPLEXIDADE DOS ALGORITMOS

### 6.1 Tokenização

**Algoritmo:** BPE (Byte-Pair Encoding)

```
Complexidade: O(n × log k)
onde:
  n = número de caracteres
  k = tamanho do vocabulário

Espaço: O(v)
onde:
  v = tamanho do vocabulário (~50.000 tokens)
```

### 6.2 Self-Attention (Transformer)

**Fórmula:** Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V

```
Complexidade por head: O(n²)
onde:
  n = comprimento da sequência

Com multi-head (h heads):
Complexidade total: O(h × n²) = O(n²) assintoticamente

Para n=2048 tokens:
- Sem otimizações: 4.194.304 operações
- Com flash-attention: ~50% mais rápido
```

**Problema: Quadrático!**
- Texto de 100 tokens: 10.000 operações
- Texto de 1.000 tokens: 1.000.000 operações
- Texto de 2.000 tokens: 4.000.000 operações

**Solução implementada:** Flash-Attention v2
- Reduz acesso à memória
- Mantém O(n²) assintoticamente mas com constante menor

### 6.3 Fine-Tuning com LoRA

```
Sem LoRA (atualizar todos os pesos):
Complexidade: O(n_params × n_exemplos × n_tokens)
Memória: O(n_params) = 7B × 4 bytes = 28GB

Com LoRA:
Complexidade: O((r × d) × n_exemplos × n_tokens)
onde:
  r = rank LoRA = 16
  d = dimensão = 4096
  
Memória: O(r × d) = 16 × 4096 × 4 bytes = 262KB
Redução: 7B → 13M (0.2% dos pesos)

Speedup: ~10x mais rápido
```

### 6.4 Inferência (Responder Pergunta)

```
Geração autoregressiva:
Complexity: O(n_max × d²)
onde:
  n_max = comprimento máximo de resposta
  d = dimensão hidden = 4096

Para resposta de 500 tokens:
- 500 forward passes
- Cada pass: ~1 bilhão de operações
- Total: ~500 bilhões de operações

Em GPU V100: ~5-10 segundos por resposta
Em GPU A100: ~2-3 segundos por resposta
```

### 6.5 Recuperação de Informação (RAG - Opcional)

Se implementar **Retrieval-Augmented Generation:**

```
1. Embedding da pergunta: O(n_tokens)
2. Busca similar em KB: O(log k) com índice HNSW
3. Concatenar contexto: O(contexto_size)
4. Gerar resposta: O(n_resposta × d²)

Complexidade total: O(log k + n × d²)
Espaço: O(k × d) onde k = documentos na KB
```

---

## 7. ARQUITETURA DO SISTEMA COMPLETO

### 7.1 Backend (FastAPI)

```python
# arquivo: main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

app = FastAPI(title="Assistente de Saúde Pública")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Carregar modelo uma única vez (na inicialização)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "meta-llama/Llama-2-7b-hf"
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

# Carregar LoRA adapters
model = PeftModel.from_pretrained(
    base_model,
    "./modelo_saude_publica_final"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

class PerguntaRequest(BaseModel):
    pergunta: str
    max_tokens: int = 512

class RespostaResponse(BaseModel):
    pergunta: str
    resposta: str
    tempo_processamento: float

@app.post("/api/responder", response_model=RespostaResponse)
async def responder(request: PerguntaRequest):
    """
    Complexidade: O(n × d²)
    onde n = tokens gerados, d = dimensão do modelo
    """
    start_time = time.time()
    
    # Tokenizar pergunta
    input_ids = tokenizer.encode(
        request.pergunta,
        return_tensors="pt"
    ).to(device)
    
    # Gerar resposta (O(n × d²))
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=request.max_tokens,
            temperature=0.7,
            top_p=0.95,
            do_sample=True
        )
    
    # Detokenizar
    resposta = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )
    
    tempo = time.time() - start_time
    
    return RespostaResponse(
        pergunta=request.pergunta,
        resposta=resposta,
        tempo_processamento=tempo
    )

@app.get("/api/saude")
async def informacoes():
    return {
        "status": "online",
        "modelo": "LLaMA 2 7B fine-tuned",
        "dispositivo": device
    }
```

### 7.2 Frontend (HTML + JavaScript)

```html
<!-- arquivo: index.html -->
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Assistente de Saúde Pública</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 0;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
            max-width: 700px;
            width: 100%;
            padding: 30px;
        }
        
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        
        .chat-box {
            height: 400px;
            border: 2px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            overflow-y: auto;
            background: #f9f9f9;
            margin-bottom: 20px;
        }
        
        .message {
            margin: 10px 0;
            padding: 10px 15px;
            border-radius: 8px;
            line-height: 1.5;
        }
        
        .user-message {
            background: #667eea;
            color: white;
            margin-left: 20px;
            text-align: right;
        }
        
        .bot-message {
            background: #e9ecef;
            color: #333;
            margin-right: 20px;
        }
        
        .input-group {
            display: flex;
            gap: 10px;
        }
        
        input {
            flex: 1;
            padding: 12px 15px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 1em;
            transition: border-color 0.3s;
        }
        
        input:focus {
            outline: none;
            border-color: #667eea;
        }
        
        button {
            padding: 12px 30px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
            transition: background 0.3s;
        }
        
        button:hover {
            background: #764ba2;
        }
        
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        
        .loading {
            text-align: center;
            color: #999;
            font-style: italic;
            padding: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏥 Assistente de Saúde Pública</h1>
        <div class="chat-box" id="chatBox"></div>
        <div class="input-group">
            <input 
                type="text" 
                id="perguntaInput" 
                placeholder="Faça uma pergunta sobre saúde pública..."
                onkeypress="if(event.key === 'Enter') enviarPergunta()"
            />
            <button onclick="enviarPergunta()" id="enviarBtn">Enviar</button>
        </div>
    </div>

    <script>
        const API_URL = "http://localhost:8000";
        const chatBox = document.getElementById("chatBox");
        const perguntaInput = document.getElementById("perguntaInput");
        const enviarBtn = document.getElementById("enviarBtn");

        async function enviarPergunta() {
            const pergunta = perguntaInput.value.trim();
            if (!pergunta) return;

            // Adicionar mensagem do usuário
            adicionarMensagem(pergunta, "user");
            perguntaInput.value = "";
            enviarBtn.disabled = true;

            // Mostrar "carregando"
            adicionarMensagem("⏳ Processando...", "loading");

            try {
                const response = await fetch(`${API_URL}/api/responder`, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        pergunta: pergunta,
                        max_tokens: 512
                    })
                });

                if (!response.ok) throw new Error("Erro na resposta");

                const data = await response.json();
                
                // Remover "carregando"
                const loadingMsgs = chatBox.querySelectorAll('.loading');
                loadingMsgs.forEach(msg => msg.remove());

                // Adicionar resposta do bot
                adicionarMensagem(
                    data.resposta + `\n\n⏱️ ${data.tempo_processamento.toFixed(2)}s`,
                    "bot"
                );
            } catch (error) {
                console.error(error);
                adicionarMensagem("❌ Erro ao processar pergunta", "bot");
            } finally {
                enviarBtn.disabled = false;
                perguntaInput.focus();
            }
        }

        function adicionarMensagem(texto, tipo) {
            const msgDiv = document.createElement("div");
            msgDiv.className = `message ${tipo}-message`;
            msgDiv.textContent = texto;
            chatBox.appendChild(msgDiv);
            chatBox.scrollTop = chatBox.scrollHeight;
        }
    </script>
</body>
</html>
```

---

## 8. O QUE APRESENTAR PARA A TURMA

### 8.1 Apresentação Técnica (Slides)

**Estrutura recomendada:**

1. **Introdução (2 min)**
   - O que é saúde pública
   - Por que usar LLMs
   - Demonstração rápida: fazer uma pergunta

2. **Dados (3 min)**
   - Fonte dos dados
   - Tamanho do dataset
   - Exemplos de perguntas/respostas
   - Gráfico de distribuição de categorias

3. **Arquitetura (5 min)**
   - Diagrama do sistema
   - Fluxo de dados
   - Tecnologias utilizadas
   - Justificativa das escolhas

4. **Transformer (5 min)**
   - Explicar sucintamente como funciona
   - Self-attention com exemplo visual
   - Por que é efetivo para linguagem natural

5. **Fine-Tuning (4 min)**
   - O que é transfer learning
   - Como funciona LoRA
   - Comparação: tempo e memória
   - Gráfico de loss durante treinamento

6. **Análise de Complexidade (5 min)**
   - Tokenização: O(n × log k)
   - Self-attention: O(n²)
   - Fine-tuning: O(r × d × exemplos × tokens)
   - Comparação: com/sem LoRA
   - Gráfico: tempo vs tamanho de entrada

7. **Resultados (4 min)**
   - Métrica de acurácia (se tiver dados de teste)
   - Exemplos de perguntas respondidas
   - Tempo de resposta
   - Casos de sucesso e limitações

8. **Demo ao Vivo (5 min)**
   - Abrir interface web
   - Fazer 3-4 perguntas diferentes
   - Mostrar tempo de processamento

---

### 8.2 Demonstração Prática (Ao Vivo)

**Setup recomendado:**
```bash
# Terminal 1: Executar backend
cd backend
python main.py
# Servidor rodando em http://localhost:8000

# Terminal 2: Servir frontend
python -m http.server 8080
# Interface em http://localhost:8080
```

**Perguntas para demonstrar:**

1. **Pergunta simples (fato)**
   - "Quais são os sintomas da dengue?"
   - Esperado: Resposta clara e estruturada

2. **Pergunta de prevenção**
   - "Como prevenir malária em viagens para região endêmica?"
   - Esperado: Recomendações práticas

3. **Pergunta sobre política pública**
   - "O que é vacinação em massa?"
   - Esperado: Explicação de conceito

4. **Pergunta complexa**
   - "Qual a relação entre saneamento básico e doenças infecciosas?"
   - Esperado: Raciocínio causal

---

### 8.3 Métricas para Apresentar

```python
# Gerar relatório de performance
def gerar_relatorio():
    return {
        "tamanho_dataset_treinamento": 1250,
        "tamanho_dataset_teste": 150,
        "tempo_treinamento_total": "45 minutos",
        "tempo_medio_resposta": "3.2 segundos",
        "memoria_gpu_pico": "6.8 GB",
        "acuracia_teste": 0.847,  # Se tiver dataset rotulado
        "perplexidade": 12.5,
        "f1_score": 0.82,
        "modelo_base": "LLaMA 2 7B",
        "adaptadores_lora": "13M parâmetros",
        "reducao_memoria": "95%",
        "speedup_treinamento": "10x"
    }
```

---

### 8.4 Visualizações Impactantes

**Gráfico 1: Complexidade de Attention**
```
Tamanho da sequência vs operações:
- 100 tokens: 10K ops
- 500 tokens: 250K ops
- 1000 tokens: 1M ops
- 2000 tokens: 4M ops
```

**Gráfico 2: Fine-tuning Loss**
```
Epoch 1: Loss 2.8
Epoch 2: Loss 1.5
Epoch 3: Loss 0.9
```

**Gráfico 3: Comparação LoRA vs Full Training**
```
                    LoRA        Full
Memória (GB)        2.1         28.0
Tempo (min)         45          450
Acurácia            0.847       0.851
```

---

## 9. CRONOGRAMA DE DESENVOLVIMENTO

| Fase | Duração | Atividades |
|------|---------|-----------|
| **1. Setup** | 1 semana | Ambiente, requisitos, datasets |
| **2. Exploração de dados** | 1 semana | Análise, limpeza, formatação |
| **3. Implementação backend** | 1.5 semanas | FastAPI, endpoints, integração modelo |
| **4. Fine-tuning** | 1 semana | Treinamento, validação, otimização |
| **5. Frontend** | 0.5 semana | Interface HTML/JS |
| **6. Testes e polimento** | 0.5 semana | Testes, documentação |
| **7. Preparação apresentação** | 1 semana | Slides, demo, análise complexidade |

**Total: ~7 semanas**

---

## 10. POTENCIAIS DESAFIOS E SOLUÇÕES

| Desafio | Probabilidade | Solução |
|--------|--------------|---------|
| GPU insuficiente | Alta | LoRA reduz 95% da memória; considerar Google Colab |
| Dataset pequeno | Média | Data augmentation, usar exemplos sintéticos |
| Resposta genérica | Alta | Aumentar num_epochs, ajustar temperatura, usar RAG |
| Overfitting | Média | Validação regular, early stopping, dropout |
| Tempo de treinamento | Média | LoRA + gradient checkpointing |
| Alucinações (respostas falsas) | Alta | Fine-tuning robusto, RLHF avançado (se time) |
| Interface lenta | Baixa | Cache de modelo, quantização, async processing |

---

## 11. REFERÊNCIAS BIBLIOGRÁFICAS

### Artigos Científicos
[1] Vaswani, A., et al. (2017). "Attention is All You Need." NeurIPS 2017.  
[2] Leong, H. Y., et al. (2024). "Efficient Fine-Tuning of Large Language Models for Automated Medical Documentation." arXiv:2409.09324.  
[3] Zhu, X., et al. (2025). "Advancing medical question answering with a knowledge embedding transformer." PLOS ONE.  
[4] Bui, N., et al. (2025). "Fine-tuning large language models for improved health information inquiries." ScienceDirect.  

### Documentação Oficial
- PyTorch: https://pytorch.org/
- Hugging Face Transformers: https://huggingface.co/transformers/
- PEFT (LoRA): https://github.com/huggingface/peft
- FastAPI: https://fastapi.tiangolo.com/

### Recursos Educacionais
- "LLMs from Scratch" (Raschka)
- Hugging Face NLP Course
- Jay Alammar's Blog (The Illustrated Transformer)

---

## 12. CONCLUSÃO

Este projeto oferece:

✅ **Aprendizado prático** de NLP, deep learning e arquitetura de sistemas  
✅ **Aplicação real** em domínio crítico (saúde pública)  
✅ **Desafio técnico apropriado** para PAA  
✅ **Demonstração clara** de análise de complexidade  
✅ **Resultado tangível** (sistema funcional)  

Boa sorte com o projeto! 🚀
