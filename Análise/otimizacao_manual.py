import matplotlib.pyplot as plt
import numpy as np
import time

# Tente importar torch, mas não falhe se não estiver disponível (ou dll faltando)
try:
    import torch
    TORCH_AVAILABLE = True
except OSError:
    print("Aviso: Falha ao carregar bibliotecas do PyTorch due a erro de DLL. A comparação com PyTorch será pulada.")
    TORCH_AVAILABLE = False
except ImportError:
    print("Aviso: PyTorch não instalado. A comparação com PyTorch será pulada.")
    TORCH_AVAILABLE = False

class OtimizadorManual:
    """
    Implementação manual de algoritmos de otimização para fins didáticos
    e cumprimento de requisitos da disciplina de PAA.
    """
    
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.history = []
        
    def gradient_descent(self, X, y, w, b, epochs):
        """
        Executa o Gradient Descent padrão (Batch).
        
        Complexidade: O(epochs * N * d)
        onde:
        - epochs: número de iterações
        - N: número de exemplos
        - d: número de features (dimensão)
        """
        N = len(y)
        loss_history = []
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Forward pass: y_pred = Xw + b
            y_pred = X.dot(w) + b
            
            # Cálculo do erro (MSE)
            error = y_pred - y
            loss = (1/N) * np.sum(error**2)
            loss_history.append(loss)
            
            # Backward pass (Cálculo dos gradientes manualmente)
            # dLoss/dw = (2/N) * X^T * error
            dw = (2/N) * X.T.dot(error)
            
            # dLoss/db = (2/N) * sum(error)
            db = (2/N) * np.sum(error)
            
            # Atualização dos parâmetros
            w = w - self.learning_rate * dw
            b = b - self.learning_rate * db
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")
                
        end_time = time.time()
        print(f"\nTempo total (Manual GD): {end_time - start_time:.4f}s")
        
        return w, b, loss_history

def demonstracao_comparativa():
    print("=== Demonstração: Otimização Manual vs PyTorch ===")
    
    # 1. Gerar dados sintéticos (Regressão Linear: y = 2x + 1 + ruído)
    np.random.seed(42)
    N = 1000
    X = 2 * np.random.rand(N, 1)
    y = 4 + 3 * X + np.random.randn(N, 1) # y = 3x + 4 + noise
    
    # --- Implementação Manual ---
    print("\n1. Executando Implementação Manual...")
    w_manual = np.random.randn(1, 1)
    b_manual = np.random.randn(1)
    
    optimizer = OtimizadorManual(learning_rate=0.1)
    w_final, b_final, loss_manual = optimizer.gradient_descent(X, y, w_manual, b_manual, epochs=1000)
    
    print(f"Resultados Manuais: w={w_final[0][0]:.2f}, b={b_final[0]:.2f}")
    
    # --- Implementação PyTorch ---
    if TORCH_AVAILABLE:
        print("\n2. Executando Implementação PyTorch...")
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).float()
        
        model = torch.nn.Linear(1, 1)
        criterion = torch.nn.MSELoss()
        sgd = torch.optim.SGD(model.parameters(), lr=0.1)
        
        loss_torch = []
        start_time = time.time()
        for epoch in range(1000):
            # Forward
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            
            # Backward & Optimize
            sgd.zero_grad()
            loss.backward()
            sgd.step()
            
            loss_torch.append(loss.item())
            
        end_time = time.time()
        print(f"Tempo total (PyTorch SGD): {end_time - start_time:.4f}s")
        
        # Comparação gráfica
        plt.figure(figsize=(10, 5))
        plt.plot(loss_manual, label='Manual Gradient Descent')
        plt.plot(loss_torch, label='PyTorch SGD', linestyle='--')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.title('Comparação de Convergência: Manual vs PyTorch')
        plt.legend()
        plt.grid(True)
        
        output_img = 'comparacao_otimizacao.png'
        plt.savefig(output_img)
        print(f"\nGráfico salvo em: {output_img}")
    else:
        print("\nSkipping PyTorch comparison due to missing library/dll.")
        # Plot only manual
        plt.figure(figsize=(10, 5))
        plt.plot(loss_manual, label='Manual Gradient Descent')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.title('Convergência Manual Gradient Descent')
        plt.legend()
        plt.grid(True)
        output_img = 'otimizacao_manual_plot.png'
        plt.savefig(output_img)
        print(f"\nGráfico salvo em: {output_img}")

if __name__ == "__main__":
    demonstracao_comparativa()
