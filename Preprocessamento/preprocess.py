import json
import re
import os
from typing import List, Dict
from transformers import AutoTokenizer

class DataProcessor:
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf"):
        """Inicializa o processador com tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception:
            print(f"⚠️ Could not load {model_name}. Using 'gpt2' as fallback for tokenization demo.")
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            
        self.max_length = 2048
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
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
    # Caminhos relativos
    input_path = "../Dataset/dataset_saude_publica.json"
    output_path = "../Dataset/dataset_processed.json"
    
    if not os.path.exists(input_path):
        print(f"❌ Arquivo {input_path} não encontrado. Execute collect_data.py primeiro.")
        exit(1)
        
    # Carregar dados
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Processar
    processor = DataProcessor()
    processados, invalidos = processor.processar_dataset(dataset)
    
    # Salvar dados processados
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processados, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 Estatísticas:")
    if processados:
        print(f"   Token count médio: {sum(p['token_count'] for p in processados) / len(processados):.0f}")
