import matplotlib.pyplot as plt

def plot_training_loss(loss_history, filename="loss_curve.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title("Curva de Aprendizado (Loss)")
    plt.xlabel("Steps/Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(filename)
    print(f"Gráfico salvo: {filename}")

def plot_comparison_bar(labels, values_a, values_b, label_a="Sem LoRA", label_b="Com LoRA", title="Comparação de Performance", filename="comparacao.png"):
    import numpy as np
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, values_a, width, label=label_a)
    rects2 = ax.bar(x + width/2, values_b, width, label=label_b)
    
    ax.set_ylabel('Valor')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    
    fig.tight_layout()
    plt.savefig(filename)
    print(f"Gráfico salvo: {filename}")

if __name__ == "__main__":
    # Teste
    loss = [10, 8, 6, 4, 3, 2, 1.5, 1.2, 1.0]
    plot_training_loss(loss)
    
    labels = ['Tempo (min)', 'Memória (GB)']
    sem_lora = [450, 24]
    com_lora = [45, 4]
    plot_comparison_bar(labels, sem_lora, com_lora)
