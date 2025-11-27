import json
import pandas as pd
import os
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

def salvar_dataset(dataset, caminho="../Dataset/dataset_saude_publica.json"):
    """
    Complexidade: O(n) onde n = número de exemplos
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    
    with open(caminho, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Dataset salvo em {caminho}")
    print(f"   Total de exemplos: {len(dataset['training_data'])}")

def carregar_dataset(caminho="../Dataset/dataset_saude_publica.json"):
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
