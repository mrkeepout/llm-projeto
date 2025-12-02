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
    
    try:
        if os.path.exists(trained_model_path):
            logger.info(f"🔄 Carregando modelo treinado de: {trained_model_path}")
            # Load base model first (assuming Llama 2)
            # Note: In a real scenario, we need the base model path. 
            # For this demo, we'll assume the adapter path contains the config or we use a default base.
            # But PeftModel needs a base_model.
            # Let's assume we use 'gpt2' as base if we can't find Llama, or just load the path as a model if it's a full save.
            # However, fine_tuning.py saves adapters.
            # We need the base model name.
            base_model_name = "meta-llama/Llama-2-7b-hf"
            
            try:
                # Configuração de Quantização apenas se tiver GPU
                quantization_config = None
                if device == "cuda":
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=False,
                    )
                
                tokenizer = AutoTokenizer.from_pretrained(base_model_name)
                
                model_kwargs = {
                    "device_map": "auto" if device == "cuda" else None,
                    "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
                    "attn_implementation": "eager", # Force standard attention to avoid FMHA kernel errors
                }
                
                if quantization_config:
                    model_kwargs["quantization_config"] = quantization_config

                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    **model_kwargs
                )
                model = PeftModel.from_pretrained(base_model, trained_model_path)
            except Exception as e:
                logger.warning(f"⚠️ Could not load Llama 2 ({e}). Falling back to GPT-2 for demo.")
                base_model_name = "gpt2"
                tokenizer = AutoTokenizer.from_pretrained(base_model_name)
                model = AutoModelForCausalLM.from_pretrained(base_model_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

        else:
            logger.warning(f"⚠️ Modelo treinado não encontrado em {trained_model_path}. Usando fallback: gpt2")
            model_name = "gpt2"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
        # Ensure pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
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

# Armazenamento de conversas em memória (Simples)
conversations = {}

# Models Pydantic
class PerguntaRequest(BaseModel):
    pergunta: str
    session_id: str = "default" # Identificador da sessão
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
    Responde pergunta sobre saúde pública com contexto.
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
        # Gerenciar histórico
        if request.session_id not in conversations:
            conversations[request.session_id] = []
        
        history = conversations[request.session_id]
        
        # Construir prompt com histórico (limitado aos últimos 3 turnos para caber no contexto)
        context_str = ""
        for q, a in history[-3:]:
            context_str += f"Human: {q}\nAssistant: {a}\n"
        
        # Prompt refinado para evitar alucinações
        prompt = f"""<s>[INST] <<SYS>>
Você é um assistente de saúde pública útil e direto. Responda apenas à pergunta feita.
Não gere novas perguntas ou diálogos. Pare de gerar texto assim que terminar a resposta.
Responda sempre em Português do Brasil.
<</SYS>>

Contexto anterior:
{context_str}

Human: {request.pergunta} [/INST]"""
        
        # Tokenizar
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(device)
        
        # Gerar resposta
        with torch.no_grad():
            output_ids = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=512,
                temperature=0.6, # Levemente reduzido para ser mais focado
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2 # Aumentado para evitar repetições
            )
        
        # Detokenizar apenas os novos tokens
        new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
        resposta = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True
        ).strip()
        
        logger.info(f"📝 Resposta RAW: {resposta}")

        # Pós-processamento inteligente para recuperar a resposta real
        
        # 1. Se o modelo gerou o papel do assistente (ex: "Assistir:", "Assistant:"), pegamos o que vem depois
        start_markers = ["Assistant:", "Assistir:", "AI:"]
        for marker in start_markers:
            if marker in resposta:
                # Pega tudo depois do marcador
                resposta = resposta.split(marker, 1)[1].strip()
                break 

        # 2. Se a resposta ainda começa com "Human:" ou "User:", tentamos pular essa linha
        # Isso acontece se o modelo repetir a pergunta do usuário antes de responder
        if resposta.startswith("Human:") or resposta.startswith("User:"):
            parts = resposta.split("\n", 1)
            if len(parts) > 1:
                resposta = parts[1].strip()
            else:
                # Se só tem a pergunta, removemos o marcador para não ficar vazio, 
                # mas provavelmente a resposta será ruim.
                resposta = resposta.replace("Human:", "").replace("User:", "").strip()

        # 3. Agora cortamos o que vier DEPOIS (alucinações de turnos futuros)
        stop_markers = ["Human:", "[/INST]", "User:", "<s>", "</s>"]
        for marker in stop_markers:
            if marker in resposta:
                resposta = resposta.split(marker)[0].strip()
        
        logger.info(f"✂️ Resposta Final: {resposta}")

        # Atualizar histórico (apenas se não for vazio)
        if resposta:
            history.append((request.pergunta, resposta))
        
        tempo_processamento = time.time() - start_time
        
        return RespostaResponse(
            pergunta=request.pergunta,
            resposta=resposta,
            tempo_processamento=tempo_processamento,
            modelo="LLaMA 2 7B + LoRA" if "llama" in str(type(model)).lower() else "GPT-2 (Demo)"
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
