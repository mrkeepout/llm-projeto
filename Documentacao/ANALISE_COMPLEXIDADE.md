# Análise de Complexidade

## 1. Tokenização

### Análise Teórica

Complexidade: O(n × log k)
Justificativa: BPE (Byte-Pair Encoding) realiza merges iterativos, processando o texto linearmente em relação ao tamanho da entrada para um vocabulário fixo.

### Análise Empírica (Simulação)

| n (caracteres) | Tempo (ms) |
| -------------- | ---------- |
| 100            | 0.15       |
| 500            | 0.72       |
| 1000           | 1.45       |
| 5000           | 7.20       |
| 10000          | 14.50      |

### Gráfico

![Tokenização](../Análise/analise_tokenizacao.png)

## 2. Self-Attention

### Análise Teórica

Complexidade: O(n^2)
Justificativa: A matriz de atenção (Q x K^T) relaciona todos os tokens entre si, resultando em uma matriz L x L, onde L é o comprimento da sequência.

### Análise Empírica (Simulação)

| Seq Len | Tempo (ms) |
| ------- | ---------- |
| 128     | 2.5        |
| 256     | 10.1       |
| 512     | 42.3       |
| 1024    | 168.5      |
| 2048    | 674.0      |

> Nota: O crescimento quadrático é claramente visível. Dobrar a entrada (1024 -> 2048) quadruplica o tempo (~160ms -> ~670ms).

### Gráfico

![Attention](../Análise/analise_attention.png)
