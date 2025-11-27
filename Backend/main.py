import time
import torch
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Carregamento global do modelo (startup)
model = None
tokenizer = None
device = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Carrega modelo na inicialização da aplicação.
    Complexidade: O(p) onde p = tamanho do modelo (7B parâmetros)
    """
    global model, tokenizer, device
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"📱 Dispositivo: {device}")
    
    # Check if trained model exists
    trained_model_path = "../Models/modelo_saude_publica_final"
    
    if os.path.exists(trained_model_path):
        model_name = trained_model_path
        logger.info(f"🔄 Carregando modelo treinado: {model_name}")
    else:
        model_name = "gpt2" # Fallback for demo if no model trained
        logger.info(f"⚠️ Modelo treinado não encontrado em {trained_model_path}. Usando fallback: {model_name}")
    
    try:
        # Carregar modelo base
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        
        # Ensure pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = base_model
        logger.info("✅ Modelo carregado com sucesso!")
        
    except Exception as e:
        logger.error(f"❌ Erro fatal ao carregar modelo: {e}")
        model = None
        
    yield
    
    # Shutdown logic if needed
    logger.info("🛑 Desligando servidor...")

# Inicializar FastAPI
app = FastAPI(
    title="🏥 Assistente de Saúde Pública",
    description="Sistema LLM para responder perguntas sobre saúde pública",
    version="1.0.0",
    lifespan=lifespan
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

@app.post("/api/responder", response_model=RespostaResponse)
async def responder(request: PerguntaRequest):
    """
    Responde pergunta sobre saúde pública.
    Complexidade: O(n × d²) onde:
    - n = tokens gerados
    - d = dimensão do modelo (4096)
    """
    start_time = time.time()
    
    if model is None:
        return RespostaResponse(
            pergunta=request.pergunta,
            resposta="Erro: Modelo não está carregado.",
            tempo_processamento=0,
            modelo="nenhum"
        )
    
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
            modelo="LLaMA 2 7B + LoRA" if "llama" in str(type(model)).lower() else "Demo Model"
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
        "modelo": "Carregado" if model else "Erro",
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
