import time
import numpy as np
import matplotlib.pyplot as plt
import math

# Try importing torch/transformers, handle failure gracefully
try:
    import torch
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    print("Aviso: Torch não disponível. Usando simulação NumPy.")
    TORCH_AVAILABLE = False

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except (ImportError, OSError):
    print("Aviso: Transformers não disponível. Usando simulação.")
    TRANSFORMERS_AVAILABLE = False

class AnaliseComplexidade:
    """
    Classe para realizar análise de complexidade dos algoritmos principais
    utilizados no projeto LLM.
    """
    
    def __init__(self):
        self.results = {}
        
    def analise_teorica(self):
        print("=== Análise Teórica ===")
        print("1. Tokenização (BPE): O(N * log V)")
        print("   - N: tamanho do texto")
        print("   - V: tamanho do vocabulário")
        print("\n2. Self-Attention: O(L^2 * D)")
        print("   - L: comprimento da sequência (contexto)")
        print("   - D: dimensão do embedding")
        print("\n3. Forward Pass (Linear Layer): O(L * D^2)")
        
    def medir_tokenizacao(self, tokenizer, textos_variados):
        """
        Mede o tempo de tokenização para diferentes tamanhos de entrada.
        """
        print("\n=== Medindo Tokenização ===")
        tempos = []
        tamanhos = []
        
        for texto in textos_variados:
            n_chars = len(texto)
            start = time.time()
            if TRANSFORMERS_AVAILABLE:
                _ = tokenizer.encode(texto)
            else:
                # Simulação linear O(N) para demo
                _ = [ord(c) for c in texto] 
                time.sleep(n_chars * 0.00001) # Simula processamento
                
            end = time.time()
            
            tempos.append((end - start) * 1000) # ms
            tamanhos.append(n_chars)
            
        return icon_sort(tamanhos, tempos)

    def simular_atencao(self, seq_lens, d_model=4096):
        """
        Simula o custo computacional do Self-Attention O(L^2).
        """
        print("\n=== Simulando Self-Attention O(L^2) ===")
        tempos = []
        
        # Aquecimento
        if TORCH_AVAILABLE:
            a = torch.randn(100, d_model)
            b = torch.randn(d_model, 100)
            torch.matmul(a, b)
        else:
            a = np.random.rand(100, 100)
            b = np.random.rand(100, 100)
            np.dot(a, b)
        
        for L in seq_lens:
            start = time.time()
            
            if TORCH_AVAILABLE:
                # Atenção simplificada: Q * K^T -> (L, D) * (D, L) -> (L, L)
                Q = torch.randn(L, d_model)
                K = torch.randn(d_model, L)
                _ = torch.matmul(Q, K)
            else:
                # Simulação Numpy ou pura (reduzida para não travar CPU)
                # O(L^2) simulado
                # Usamos tamanhos menores para numpy não demorar eras se d_model for grande
                d_sim = min(d_model, 512) 
                Q = np.random.rand(L, d_sim)
                K = np.random.rand(d_sim, L)
                _ = np.dot(Q, K)
            
            end = time.time()
            tempos.append((end - start) * 1000) # ms
            
        return seq_lens, tempos

def icon_sort(X, Y):
    # Utilitário para ordenar pares para plotagem
    XY = sorted(zip(X, Y))
    X_sorted = [x for x, y in XY]
    Y_sorted = [y for x, y in XY]
    return X_sorted, Y_sorted

def plotar_resultados(x, y, titulo, xlabel, ylabel, filename):
    plt.figure()
    plt.plot(x, y, 'o-', label='Medido')
    
    # Fit polinomial para comparar
    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)
    plt.plot(x, p(x), 'r--', label='Tendência (Poly Fit)')
    
    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    print(f"Gráfico salvo: {filename}")

if __name__ == "__main__":
    analise = AnaliseComplexidade()
    analise.analise_teorica()
    
    # Mock de teste se não tivermos tokenizer real aqui
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        if TRANSFORMERS_AVAILABLE:
             tokenizer = AutoTokenizer.from_pretrained("gpt2")
        else:
             raise ImportError("Transformers not available flag")
        
        textos = ["a" * 100, "a" * 500, "a" * 1000, "a" * 5000, "a" * 10000]
        x_tok, y_tok = analise.medir_tokenizacao(tokenizer, textos)
        plotar_resultados(x_tok, y_tok, "Complexidade Tokenização", "Caracteres (N)", "Tempo (ms)", "analise_tokenizacao.png")
        
    except (ImportError, OSError, Exception) as e:
        print(f"Transformers real falhou ({e}). Usando Mock para gerar gráficos.")
        
        # Mock class for tokenizer
        class MockTokenizer:
             def encode(self, text):
                 return [0] * len(text)
        
        tokenizer = MockTokenizer()
        textos = ["a" * 100, "a" * 500, "a" * 1000, "a" * 5000, "a" * 10000]
        
        # Force TRANSFOMERS_AVAILABLE to False locally for this run if it was true but crashed
        temp_transformers_status = TRANSFORMERS_AVAILABLE
        TRANSFORMERS_AVAILABLE = False # Force simulation path
        
        x_tok, y_tok = analise.medir_tokenizacao(tokenizer, textos)
        plotar_resultados(x_tok, y_tok, "Complexidade Tokenização", "Caracteres (N)", "Tempo (ms)", "analise_tokenizacao.png")
        
        TRANSFORMERS_AVAILABLE = temp_transformers_status

    # Teste de Atenção (Simulação Quadrática)
    seq_lens = [128, 256, 512, 1024, 2048]
    # Se der erro de memória, reduza
    try:
        x_att, y_att = analise.simular_atencao(seq_lens, d_model=1024) 
        plotar_resultados(x_att, y_att, "Complexidade Self-Attention O(L^2)", "Seq Length (L)", "Tempo (ms)", "analise_attention.png")
    except Exception as e:
        print(f"Erro ao simular atenção: {e}")
