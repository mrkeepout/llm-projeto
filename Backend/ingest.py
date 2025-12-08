import os
import json
import logging
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurações
PERSIST_DIRECTORY = "../db_chroma"
DATASET_PATH = "../Dataset/dataset_medquad.json"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def ingest_data():
    """
    Lê o dataset, cria chunks e salva no ChromaDB.
    """
    logger.info("🚀 Iniciando ingestão de dados...")

    # 1. Carregar dados
    if not os.path.exists(DATASET_PATH):
        logger.error(f"❌ Arquivo {DATASET_PATH} não encontrado.")
        return

    logger.info(f"📂 Lendo arquivo {DATASET_PATH}...")
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extrair textos (assumindo formato MedQuad ou lista simples)
    documents = []
    
    # Se for o formato MedQuad (lista de dicts)
    if isinstance(data, list):
        for item in data:
            # Combinar pergunta e resposta para contexto
            content = f"Pergunta: {item.get('pergunta', '')}\nResposta: {item.get('resposta', '')}"
            meta = {"source": "MedQuad", "id": item.get('id', 'unknown')}
            documents.append(Document(page_content=content, metadata=meta))
    elif isinstance(data, dict) and 'training_data' in data:
         for item in data['training_data']:
            content = f"Pergunta: {item.get('pergunta', '')}\nResposta: {item.get('resposta', '')}"
            meta = {"source": "MedQuad", "id": item.get('id', 'unknown')}
            documents.append(Document(page_content=content, metadata=meta))

    logger.info(f"📄 Carregados {len(documents)} documentos brutos.")

    # 2. Splitter (Dividir em chunks menores)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    texts = text_splitter.split_documents(documents)
    logger.info(f"🧩 Dividido em {len(texts)} chunks.")

    # 3. Criar Embeddings e VectorDB
    logger.info(f"🧠 Carregando modelo de embedding: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    logger.info("💾 Criando/Atualizando banco de dados ChromaDB...")
    # Cria o DB e persiste automaticamente no diretório especificado
    db = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    
    # db.persist() # Em versões novas do Chroma, persist é automático ou não necessário explicitamente dependendo da config, mas o diretório garante persistência.
    
    logger.info(f"✅ Ingestão concluída! Banco salvo em {PERSIST_DIRECTORY}")

if __name__ == "__main__":
    ingest_data()
