# Guia Prático: Implementação e Código

---

## PARTE 1: INSTALAÇÃO E SETUP

### 1.1 Criar Ambiente Virtual

```bash
# Criar diretório do projeto
mkdir projeto_llm_saude
cd projeto_llm_saude

# Criar ambiente virtual
python3.10 -m venv venv

# Ativar (Linux/Mac)
source venv/bin/activate

# Ativar (Windows)
venv\Scripts\activate

# Atualizar pip
pip install --upgrade pip
```

### 1.2 Instalar Dependências

```bash
# Salvar em requirements.txt
cat > requirements.txt << EOF
torch>=2.0.0
transformers>=4.35.0
peft>=0.7.0
pydantic>=2.0.0
fastapi>=0.104.0
uvicorn>=0.24.0
numpy>=1.24.0
scikit-learn>=1.3.0
pandas>=2.0.0
python-dotenv>=1.0.0
EOF

# Instalar
pip install -r requirements.txt
```

---

## PARTE 2: PREPARAÇÃO DE DADOS

### 2.1 Script de Coleta de Dados (collect_data.py)

```python
import json
import pandas as pd
from pathlib import Path

def criar_dataset_exemplo():
    """
    Cria um dataset de exemplo para testes rápidos.
    Em produção, seria de fontes reais.
    """
    
    dataset = {
        "training_data": [
            {
                "id": 1,
                "pergunta": "O que é dengue?",
                "resposta": "Dengue é uma doença infecciosa causada pelo vírus da dengue, transmitido principalmente pelo mosquito Aedes aegypti. É uma das doenças mais importantes de saúde pública nas regiões tropicais e subtropicais.",
                "categoria": "doenças_infecciosas",
                "confiança": 0.95
            },
            {
                "id": 2,
                "pergunta": "Quais são os sintomas da dengue?",
                "resposta": "Os sintomas principais incluem: febre alta (até 40°C), dor de cabeça intensa, dor nos olhos, músculos e articulações, fraqueza e rash cutâneo. Os sintomas geralmente aparecem entre 3-14 dias após a infecção.",
                "categoria": "sintomas",
                "confiança": 0.98
            },
            {
                "id": 3,
                "pergunta": "Como se transmite o vírus Zika?",
                "resposta": "O vírus Zika é transmitido principalmente pelo mosquito Aedes aegypti infectado, assim como a dengue. Também pode ser transmitido sexualmente e durante a gravidez (de mãe para filho).",
                "categoria": "transmissão",
                "confiança": 0.96
            },
            {
                "id": 4,
                "pergunta": "Qual é a diferença entre COVID-19 e gripe comum?",
                "resposta": "COVID-19 é causada pelo vírus SARS-CoV-2 e é mais grave que a gripe. Ambas são respiratórias, mas COVID-19 pode levar a complicações mais sérias como pneumonia grave, trombose e síndrome do desconforto respiratório agudo.",
                "categoria": "diferenciais",
                "confiança": 0.94
            },
            {
                "id": 5,
                "pergunta": "Como funciona uma vacina?",
                "resposta": "Uma vacina funciona estimulando o sistema imunológico a reconhecer e combater patógenos específicos sem causar a doença. Ela contém antígenos que treinam o sistema imune a produzir anticorpos e células de memória.",
                "categoria": "vacinação",
                "confiança": 0.97
            },
            # ... adicionar mais exemplos
        ]
    }
    
    return dataset

def salvar_dataset(dataset, caminho="dataset_saude_publica.json"):
    """
    Complexidade: O(n) onde n = número de exemplos
    """
    with open(caminho, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Dataset salvo em {caminho}")
    print(f"   Total de exemplos: {len(dataset['training_data'])}")

def carregar_dataset(caminho="dataset_saude_publica.json"):
    """
    Carrega dataset de arquivo JSON.
    Complexidade: O(n)
    """
    with open(caminho, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    return dataset

def analisar_dataset(dataset):
    """
    Analisa características do dataset.
    Complexidade: O(n × m) onde m = comprimento médio do texto
    """
    df = pd.DataFrame(dataset['training_data'])
    
    print("\n📊 ANÁLISE DO DATASET")
    print("=" * 50)
    print(f"Total de exemplos: {len(df)}")
    print(f"\nDistribuição por categoria:")
    print(df['categoria'].value_counts())
    
    print(f"\nComprimento médio das perguntas: {df['pergunta'].str.len().mean():.0f} caracteres")
    print(f"Comprimento médio das respostas: {df['resposta'].str.len().mean():.0f} caracteres")
    
    print(f"\nConfiança média: {df['confiança'].mean():.2%}")

if __name__ == "__main__":
    # Criar dataset de exemplo
    dataset = criar_dataset_exemplo()
    
    # Salvar
    salvar_dataset(dataset)
    
    # Analisar
    analisar_dataset(dataset)
```

### 2.2 Script de Pré-processamento (preprocess.py)

```python
import json
import re
from typing import List, Dict
from transformers import AutoTokenizer

class DataProcessor:
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf"):
        """Inicializa o processador com tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = 2048
    
    def limpar_texto(self, texto: str) -> str:
        """
        Remove caracteres especiais e normaliza.
        Complexidade: O(n) onde n = comprimento do texto
        """
        # Remover URLs
        texto = re.sub(r'http\S+|www.\S+', '', texto)
        
        # Remover caracteres especiais desnecessários
        texto = re.sub(r'[^\w\s\.\,\!\?\-\:\;]', '', texto)
        
        # Remover espaços extras
        texto = re.sub(r'\s+', ' ', texto).strip()
        
        return texto
    
    def normalizar_texto(self, texto: str) -> str:
        """
        Normaliza e padroniza.
        Complexidade: O(n)
        """
        # Converter para lowercase (opcional para domínio médico)
        # texto = texto.lower()
        
        # Remover pontuação redundante
        texto = re.sub(r'\.+', '.', texto)
        
        return texto
    
    def processar_exemplo(self, exemplo: Dict) -> Dict:
        """
        Processa um único exemplo.
        Complexidade: O(m) onde m = comprimento do texto
        """
        pergunta = self.limpar_texto(exemplo['pergunta'])
        pergunta = self.normalizar_texto(pergunta)
        
        resposta = self.limpar_texto(exemplo['resposta'])
        resposta = self.normalizar_texto(resposta)
        
        # Criar prompt no formato esperado
        prompt = f"Pergunta: {pergunta}\nResposta: {resposta}"
        
        # Tokenizar
        tokens = self.tokenizer.encode(
            prompt,
            max_length=self.max_length,
            truncation=True
        )
        
        # Verificar se é válido
        is_valid = (
            len(tokens) >= 20 and  # Mínimo de tokens
            len(tokens) <= self.max_length and  # Máximo de tokens
            len(pergunta) > 5 and  # Pergunta com sentido
            len(resposta) > 10  # Resposta com sentido
        )
        
        return {
            'id': exemplo['id'],
            'pergunta': pergunta,
            'resposta': resposta,
            'prompt': prompt,
            'tokens': tokens,
            'token_count': len(tokens),
            'is_valid': is_valid,
            'categoria': exemplo.get('categoria', 'indefinido')
        }
    
    def processar_dataset(self, dataset: Dict) -> tuple:
        """
        Processa todo o dataset.
        Complexidade: O(n × m) onde n = exemplos, m = comprimento médio
        """
        processados = []
        inválidos = []
        
        for exemplo in dataset['training_data']:
            exemplo_proc = self.processar_exemplo(exemplo)
            
            if exemplo_proc['is_valid']:
                processados.append(exemplo_proc)
            else:
                inválidos.append(exemplo)
        
        print(f"✅ {len(processados)} exemplos válidos")
        print(f"❌ {len(inválidos)} exemplos rejeitados")
        
        return processados, inválidos

if __name__ == "__main__":
    from collect_data import criar_dataset_exemplo
    
    # Carregar dados
    dataset = criar_dataset_exemplo()
    
    # Processar
    processor = DataProcessor()
    processados, invalidos = processor.processar_dataset(dataset)
    
    # Salvar dados processados
    with open("dataset_processed.json", 'w') as f:
        json.dump(processados, f, indent=2)
    
    print(f"\n📊 Estatísticas:")
    print(f"   Token count médio: {sum(p['token_count'] for p in processados) / len(processados):.0f}")
```

---

## PARTE 3: TREINAMENTO (fine-tuning.py)

```python
import torch
import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset

class LLMTrainerSaudePública:
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf"):
        """
        Complexidade: O(1) - carrega arquivos pré-existentes
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"📱 Dispositivo: {self.device}")
        print(f"🤖 Modelo: {model_name}")
        
        # Carregar modelo e tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        # Adicionar padding token se não existir
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def aplicar_lora(self):
        """
        Aplica LoRA para fine-tuning eficiente.
        Complexidade: O(p) onde p = número de parâmetros LoRA
        """
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Imprimir informação dos parâmetros
        self.model.print_trainable_parameters()
        
        return self.model
    
    def preparar_dados(self, dataset_path: str):
        """
        Prepara dados para treinamento.
        Complexidade: O(n × m) onde n = exemplos, m = comprimento médio
        """
        # Carregar dados
        with open(dataset_path, 'r') as f:
            dados = json.load(f)
        
        # Se for lista de dicts processados
        if isinstance(dados, list):
            prompts = [d['prompt'] for d in dados]
        else:
            prompts = [d['prompt'] for d in dados['training_data']]
        
        # Tokenizar
        def tokenize(texto):
            return self.tokenizer(
                texto,
                max_length=2048,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
        
        # Criar dataset
        dataset = Dataset.from_dict({'text': prompts})
        dataset = dataset.map(
            lambda x: tokenize(x['text']),
            batched=True
        )
        
        # Split train/eval
        split_data = dataset.train_test_split(test_size=0.1)
        
        return split_data['train'], split_data['test']
    
    def treinar(self, train_dataset, eval_dataset):
        """
        Executa fine-tuning com LoRA.
        Complexidade: O(epochs × len(dataset) × forward_pass)
        Típico: ~45 minutos em GPU V100 para 1000 exemplos
        """
        training_args = TrainingArguments(
            output_dir="./modelo_saude_publica",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            learning_rate=2e-4,
            optim="adamw_8bit"
        )
        
        data_collator = DataCollatorForLanguageModeling(
            self.tokenizer,
            mlm=False
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )
        
        # Iniciar treinamento
        print("\n🚀 Iniciando treinamento...")
        trainer.train()
        
        # Salvar modelo
        self.model.save_pretrained("./modelo_saude_publica_final")
        self.tokenizer.save_pretrained("./modelo_saude_publica_final")
        
        print("✅ Modelo salvo!")
        
        return trainer

if __name__ == "__main__":
    # Inicializar trainer
    trainer = LLMTrainerSaudePública()
    
    # Aplicar LoRA
    trainer.aplicar_lora()
    
    # Preparar dados
    train_data, eval_data = trainer.preparar_dados("dataset_processed.json")
    
    # Treinar
    trainer.treinar(train_data, eval_data)
```

---

## PARTE 4: SERVIDOR BACKEND (main.py)

```python
import time
import torch
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="🏥 Assistente de Saúde Pública",
    description="Sistema LLM para responder perguntas sobre saúde pública",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models Pydantic
class PerguntaRequest(BaseModel):
    pergunta: str
    max_tokens: int = 512
    temperature: float = 0.7

class RespostaResponse(BaseModel):
    pergunta: str
    resposta: str
    tempo_processamento: float
    modelo: str

# Carregamento global do modelo (startup)
model = None
tokenizer = None
device = None

@app.on_event("startup")
async def load_model():
    """
    Carrega modelo na inicialização da aplicação.
    Complexidade: O(p) onde p = tamanho do modelo (7B parâmetros)
    """
    global model, tokenizer, device
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"📱 Dispositivo: {device}")
    
    model_name = "meta-llama/Llama-2-7b-hf"
    logger.info(f"🔄 Carregando modelo base: {model_name}")
    
    # Carregar modelo base
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        load_in_8bit=True if device == "cuda" else False
    )
    
    # Carregar LoRA adapters
    try:
        model = PeftModel.from_pretrained(
            base_model,
            "./modelo_saude_publica_final"
        )
        logger.info("✅ LoRA adapters carregados!")
    except:
        logger.warning("⚠️ Adapters não encontrados, usando modelo base")
        model = base_model

@app.post("/api/responder", response_model=RespostaResponse)
async def responder(request: PerguntaRequest):
    """
    Responde pergunta sobre saúde pública.
    Complexidade: O(n × d²) onde:
    - n = tokens gerados
    - d = dimensão do modelo (4096)
    """
    start_time = time.time()
    
    try:
        # Preparar prompt
        prompt = f"Pergunta: {request.pergunta}\nResposta:"
        
        # Tokenizar
        input_ids = tokenizer.encode(
            prompt,
            return_tensors="pt"
        ).to(device)
        
        # Gerar resposta
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Detokenizar
        resposta_completa = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )
        
        # Extrair apenas a resposta (sem prompt)
        resposta = resposta_completa.replace(prompt, "").strip()
        
        tempo_processamento = time.time() - start_time
        
        return RespostaResponse(
            pergunta=request.pergunta,
            resposta=resposta,
            tempo_processamento=tempo_processamento,
            modelo="LLaMA 2 7B + LoRA"
        )
    
    except Exception as e:
        logger.error(f"❌ Erro: {str(e)}")
        return RespostaResponse(
            pergunta=request.pergunta,
            resposta=f"Desculpe, ocorreu um erro: {str(e)}",
            tempo_processamento=time.time() - start_time,
            modelo="erro"
        )

@app.get("/api/saude")
async def status():
    """Retorna status do servidor"""
    return {
        "status": "online",
        "modelo": "LLaMA 2 7B + LoRA",
        "dispositivo": device,
        "tema": "Saúde Pública"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "mensagem": "Bem-vindo ao Assistente de Saúde Pública!",
        "docs": "/docs",
        "endpoints": {
            "responder": "POST /api/responder",
            "status": "GET /api/saude"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
```

---

## PARTE 5: INTERFACE WEB (index.html)

```html
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🏥 Assistente de Saúde Pública</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            max-width: 800px;
            width: 100%;
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2em;
            margin-bottom: 10px;
        }
        
        .header p {
            opacity: 0.9;
            font-size: 0.9em;
        }
        
        .chat-container {
            height: 450px;
            overflow-y: auto;
            padding: 20px;
            background: #f8f9fa;
        }
        
        .message {
            margin: 15px 0;
            display: flex;
            animation: fadeIn 0.3s ease-in;
        }
        
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .message.user {
            justify-content: flex-end;
        }
        
        .message.bot {
            justify-content: flex-start;
        }
        
        .message-content {
            max-width: 70%;
            padding: 12px 16px;
            border-radius: 10px;
            line-height: 1.5;
            word-wrap: break-word;
        }
        
        .user .message-content {
            background: #667eea;
            color: white;
        }
        
        .bot .message-content {
            background: #e9ecef;
            color: #333;
        }
        
        .loading {
            display: flex;
            gap: 5px;
            padding: 12px 16px;
        }
        
        .loading span {
            width: 8px;
            height: 8px;
            background: #667eea;
            border-radius: 50%;
            animation: bounce 1.4s infinite;
        }
        
        .loading span:nth-child(2) {
            animation-delay: 0.2s;
        }
        
        .loading span:nth-child(3) {
            animation-delay: 0.4s;
        }
        
        @keyframes bounce {
            0%, 80%, 100% { transform: translateY(0); opacity: 0.6; }
            40% { transform: translateY(-10px); opacity: 1; }
        }
        
        .input-area {
            padding: 20px;
            background: white;
            border-top: 1px solid #ddd;
            display: flex;
            gap: 10px;
        }
        
        input {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 1em;
            transition: all 0.3s;
        }
        
        input:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        button {
            padding: 12px 30px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s;
        }
        
        button:hover {
            background: #764ba2;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(118, 75, 162, 0.3);
        }
        
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        
        .timestamp {
            font-size: 0.75em;
            opacity: 0.7;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 Assistente de Saúde Pública</h1>
            <p>Faça suas perguntas sobre saúde pública e receba respostas baseadas em IA</p>
        </div>
        
        <div class="chat-container" id="chatContainer"></div>
        
        <div class="input-area">
            <input
                type="text"
                id="perguntaInput"
                placeholder="Digite sua pergunta sobre saúde pública..."
                onkeypress="if(event.key === 'Enter') enviarPergunta()"
            />
            <button onclick="enviarPergunta()" id="enviarBtn">Enviar</button>
        </div>
    </div>

    <script>
        const API_URL = "http://localhost:8000";
        const chatContainer = document.getElementById("chatContainer");
        const perguntaInput = document.getElementById("perguntaInput");
        const enviarBtn = document.getElementById("enviarBtn");

        function formatarHora() {
            const agora = new Date();
            return agora.toLocaleTimeString('pt-BR', {
                hour: '2-digit',
                minute: '2-digit'
            });
        }

        function adicionarMensagem(texto, tipo, tempo = null) {
            const msgDiv = document.createElement("div");
            msgDiv.className = `message ${tipo}`;
            
            const content = document.createElement("div");
            content.className = "message-content";
            content.textContent = texto;
            
            msgDiv.appendChild(content);
            
            if (tempo !== null) {
                const timestamp = document.createElement("div");
                timestamp.className = "timestamp";
                timestamp.textContent = `${tempo}ms`;
                msgDiv.appendChild(timestamp);
            }
            
            chatContainer.appendChild(msgDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        function mostrarCarregando() {
            const msgDiv = document.createElement("div");
            msgDiv.className = "message bot";
            
            const content = document.createElement("div");
            content.className = "loading";
            content.innerHTML = "<span></span><span></span><span></span>";
            
            msgDiv.appendChild(content);
            chatContainer.appendChild(msgDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            
            return msgDiv;
        }

        async function enviarPergunta() {
            const pergunta = perguntaInput.value.trim();
            if (!pergunta) return;

            // Adicionar mensagem do usuário
            adicionarMensagem(pergunta, "user");
            perguntaInput.value = "";
            enviarBtn.disabled = true;

            // Mostrar indicador de carregamento
            const loadingMsg = mostrarCarregando();

            try {
                const response = await fetch(`${API_URL}/api/responder`, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        pergunta: pergunta,
                        max_tokens: 512,
                        temperature: 0.7
                    })
                });

                if (!response.ok) throw new Error("Erro na resposta do servidor");

                const data = await response.json();
                
                // Remover indicador de carregamento
                loadingMsg.remove();

                // Adicionar resposta do bot
                const tempoMs = Math.round(data.tempo_processamento * 1000);
                adicionarMensagem(data.resposta, "bot", tempoMs);

            } catch (error) {
                console.error(error);
                loadingMsg.remove();
                adicionarMensagem(
                    "❌ Desculpe, houve um erro ao processar sua pergunta. Tente novamente.",
                    "bot"
                );
            } finally {
                enviarBtn.disabled = false;
                perguntaInput.focus();
            }
        }

        // Mensagem inicial
        window.addEventListener('load', () => {
            adicionarMensagem(
                "Olá! Sou um assistente de IA especializado em saúde pública. Faça suas perguntas!",
                "bot"
            );
        });
    </script>
</body>
</html>
```

---

## PARTE 6: SCRIPT DE EXECUÇÃO (run.sh)

```bash
#!/bin/bash

echo "🚀 Iniciando Assistente de Saúde Pública"
echo "========================================"

# Verificar Python
python3 --version

# Ativar ambiente virtual
source venv/bin/activate

# Executar servidor
echo "🌐 Iniciando servidor backend na porta 8000..."
python main.py
```

---

## EXECUÇÃO DO PROJETO

```bash
# 1. Setup inicial
chmod +x run.sh
source venv/bin/activate
pip install -r requirements.txt

# 2. Preparar dados
python collect_data.py
python preprocess.py

# 3. Treinar modelo
python fine_tuning.py

# 4. Executar servidor (Terminal 1)
python main.py

# 5. Acessar interface (Terminal 2)
# Abrir arquivo index.html no navegador
# ou servir com: python -m http.server 8080
```

---

Boa sorte! 🎓

