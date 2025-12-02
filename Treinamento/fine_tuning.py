import torch
import json
import os
import random
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
        
        try:
            # Configuração de Quantização (4-bit) apenas se tiver GPU
            quantization_config = None
            if self.device == "cuda":
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=False,
                )

            # Carregar modelo e tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            
            model_kwargs = {
                "device_map": "auto" if self.device == "cuda" else None,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "attn_implementation": "eager", # Force standard attention to avoid FMHA kernel errors
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            # Preparar modelo para treinamento em k-bit se estiver usando quantização
            if quantization_config:
                from peft import prepare_model_for_kbit_training
                self.model = prepare_model_for_kbit_training(self.model)
            
        except Exception as e:
            print(f"❌ Erro ao carregar modelo {model_name}: {e}")
            print("⚠️ Mudando para 'gpt2' para teste (sem necessidade de GPU/Token)...")
            self.model_name = "gpt2"
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.model = AutoModelForCausalLM.from_pretrained("gpt2")
            # GPT2 não tem pad token por padrão
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Garantir pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Fix for gpt2
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
    
    def aplicar_lora(self):
        """
        Aplica LoRA para fine-tuning eficiente.
        Complexidade: O(p) onde p = número de parâmetros LoRA
        """
        # Adjust target modules based on model type
        if "llama" in self.model_name.lower():
            target_modules = ["q_proj", "v_proj"]
        else:
            # For GPT2 or others (demo fallback)
            target_modules = ["c_attn"]

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Imprimir informação dos parâmetros
        self.model.print_trainable_parameters()
        
        return self.model
    
    def preparar_dados(self, dataset_path: str, limit: int = None):
        """
        Prepara dados para treinamento.
        Complexidade: O(n × m) onde n = exemplos, m = comprimento médio
        """
        # Carregar dados
        if not os.path.exists(dataset_path):
            print(f"⚠️ Arquivo {dataset_path} não encontrado. Criando dados dummy.")
            dados = [{"prompt": "Pergunta: Teste\nResposta: Teste"}] * 10
        else:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dados = json.load(f)
        
        # Se for lista de dicts processados
        if isinstance(dados, list):
            prompts = [d['prompt'] for d in dados]
        else:
            prompts = [d['prompt'] for d in dados['training_data']]
            
        # Aplicar limite se solicitado
        if limit:
            print(f"⚠️ Limitando dados a {limit} exemplos para teste rápido.")
            random.shuffle(prompts)
            prompts = prompts[:limit]
        
        print(f"📊 Carregados {len(prompts)} exemplos para treinamento.")
        
        # Tokenizar
        def tokenize(texto):
            return self.tokenizer(
                texto,
                max_length=512, # Reduced for demo speed
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
        if len(dataset) > 1:
            split_data = dataset.train_test_split(test_size=0.1)
            return split_data['train'], split_data['test']
        else:
            return dataset, dataset
    
    def treinar(self, train_dataset, eval_dataset):
        """
        Executa fine-tuning com LoRA.
        Complexidade: O(epochs × len(dataset) × forward_pass)
        Típico: ~45 minutos em GPU V100 para 1000 exemplos
        """
        output_dir = "../Models/modelo_saude_publica"
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=1, # Reduced for demo
            per_device_train_batch_size=2, # Reduced for compatibility
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=1,
            eval_strategy="no", # Disable eval for speed in demo
            save_strategy="no",
            load_best_model_at_end=False,
            learning_rate=2e-4,
            use_cpu=not torch.cuda.is_available()
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
        final_path = "../Models/modelo_saude_publica_final"
        self.model.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)
        
        print(f"✅ Modelo salvo em {final_path}!")
        
        return trainer

if __name__ == "__main__":
    # Inicializar trainer
    trainer = LLMTrainerSaudePública()
    
    # Aplicar LoRA
    trainer.aplicar_lora()
    
    # Preparar dados (usando o novo dataset MedQuad)
    # Adicione limit=100 para teste rápido, ou remova para treinar com tudo
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "../Dataset/dataset_medquad.json")
    train_data, eval_data = trainer.preparar_dados(dataset_path, limit=5000)
    
    # Treinar
    trainer.treinar(train_data, eval_data)
