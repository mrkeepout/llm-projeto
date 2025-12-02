import csv
import json
import os

def convert_medquad_to_json(csv_path, json_path):
    data = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for i, row in enumerate(reader):
                # Map CSV columns to JSON fields
                # CSV: question, answer, source, focus_area
                # JSON: id, pergunta, resposta, categoria, confiança
                
                item = {
                    "id": i + 1,
                    "pergunta": row.get('question', '').strip(),
                    "resposta": row.get('answer', '').strip(),
                    "categoria": row.get('focus_area', 'General'),
                    "confiança": 1.0,  # High confidence for curated medical dataset
                    "prompt": f"Pergunta: {row.get('question', '').strip()}\nResposta: {row.get('answer', '').strip()}"
                }
                
                # Only add if question and answer are present
                if item['pergunta'] and item['resposta']:
                    data.append(item)
                    
        output = {
            "training_data": data
        }
        
        with open(json_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(output, jsonfile, ensure_ascii=False, indent=2)
            
        print(f"Successfully converted {len(data)} items from {csv_path} to {json_path}")
        
    except Exception as e:
        print(f"Error converting file: {e}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(base_dir, 'medquad.csv')
    json_file = os.path.join(base_dir, 'dataset_medquad.json')
    
    if os.path.exists(csv_file):
        convert_medquad_to_json(csv_file, json_file)
    else:
        print(f"File not found: {csv_file}")
