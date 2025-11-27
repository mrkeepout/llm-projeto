# CHECKLIST E GUIA DE ENTREGA FINAL

**Projeto:** Sistema LLM para Responder Perguntas sobre Saúde Pública  
**Disciplina:** Projeto e Análise de Algoritmos (PAA)  
**Data de Entrega:** 5 de novembro de 2025 (23:55)

---

## FASE 1: DOCUMENTAÇÃO E PLANEJAMENTO ✓

### Documento Técnico (PDF Obrigatório)

O documento PDF que será entregue deve conter:

#### Seção 1: Tema e Contexto (1 página)
- [ ] Nome do tema: "Questões de Saúde Pública com LLM"
- [ ] Justificativa: Por que escolheu este tema?
- [ ] Impacto potencial: Como ajuda a comunidade?
- [ ] Relevância para PAA: Como envolve algoritmos eficientes?

#### Seção 2: Integrantes do Grupo (0.5 página)
- [ ] Nome completo de cada integrante
- [ ] Matrícula de cada integrante
- [ ] Email de contato
- [ ] Divisão de responsabilidades

#### Seção 3: Linguagens Utilizadas (0.5 página)
- [ ] **Backend:** Python 3.10+ (Linguagem principal)
- [ ] **Frontend:** HTML5, CSS3, JavaScript (ES6+)
- [ ] Justificativa de cada escolha

#### Seção 4: Stack Tecnológico (2 páginas)
Complete a tabela:

| Componente | Tecnologia | Licença | Propósito |
|-----------|-----------|---------|----------|
| Framework ML | PyTorch 2.0+ | BSD | Treinamento deep learning |
| Biblioteca NLP | HF Transformers | Apache 2.0 | Modelos de linguagem |
| Fine-tuning | PEFT (LoRA) | Apache 2.0 | Treinamento eficiente |
| Tokenização | HF Tokenizers | Apache 2.0 | Processamento de texto |
| Servidor | FastAPI | MIT | API REST backend |
| Servidor Web | Uvicorn | BSD | Servidor ASGI |
| Modelo Base | LLaMA 2 7B | Llama Community | Modelo pré-treinado |
| Interface | HTML5/JS | MIT | Apresentação frontend |
| Banco de Dados | SQLite3 | PD | Persistência (opcional) |

**Importante:** Todas as licenças DEVEM estar na lista aprovada!

#### Seção 5: Algoritmos e Análise de Complexidade (3-4 páginas)

Descrever cada algoritmo principal:

**5.1 Tokenização (BPE)**
```
Algoritmo: Byte-Pair Encoding

Entrada: texto bruto
Saída: sequência de tokens

Complexidade: O(n × log k)
Espaço: O(v)

Justificativa:
- n = caracteres do texto
- k = tamanho do vocabulário
- v = vocab final

Exemplo prático:
"Qual é o período de incubação?"
→ [1234, 5678, 9012, ...]
```

**5.2 Self-Attention (Transformer)**
```
Algoritmo: Multi-Head Self-Attention

Fórmula: Attention(Q,K,V) = softmax(QK^T/√d_k)V

Complexidade por head: O(n²)
Espaço: O(n × d)

Análise:
- n = comprimento sequência
- d = dimensão do modelo
- Com h heads em paralelo: O(h×n²) ≈ O(n²)

Problema: Quadrático!
Solução: Flash-Attention (reduz acesso memória)

Exemplo numérico:
n=2048 (máximo tokens)
Operações: 2048² = 4.194.304 por head
Com 8 heads: ~33M operações
```

**5.3 Feed-Forward Network**
```
Algoritmo: MLP (Multilayer Perceptron)

Complexidade por exemplo: O(n × d²)
Espaço: O(d²)

Análise:
d = 4096 (dimensão)
d² = 16.777.216 parâmetros

Com n tokens:
n × d² operações por forward pass
```

**5.4 Fine-tuning com LoRA**
```
Algoritmo: Low-Rank Adaptation

Complexidade: O(E × N × r × d)

Parâmetros:
- E = número de épocas (3)
- N = número de exemplos (1250)
- r = rank LoRA (16)
- d = dimensão (4096)

Total: 3 × 1250 × 16 × 4096 = 245M operações
(vs 3 × 1250 × 7B sem LoRA)

Redução: 99.7% menos computação!
```

**5.5 Inferência (Geração)**
```
Algoritmo: Geração Autoregressiva

Complexidade: O(m × n² × d)

Parâmetros:
- m = tokens gerados (até 512)
- n = contexto (até 2048)
- d = dimensão (4096)

Análise de tempo:
Cada token = 1 forward pass
512 tokens = 512 forward passes
Cada pass = ~1 segundo
Total = ~500 segundos no máximo
Com otimizações: 3-5 segundos típico
```

#### Seção 6: Metodologia de Treinamento (2 páginas)

**6.1 Dataset**
```
Fonte: [especificar suas fontes reais]
Tamanho: 1250 exemplos (75% train, 25% test)
Distribuição: [descrever categorias]
Qualidade: [descrever processo de validação]
```

**6.2 Configuração de Fine-tuning**
```
Modelo Base: LLaMA 2 7B

Configuração LoRA:
- rank: 16
- lora_alpha: 32
- dropout: 0.05
- target_modules: ["q_proj", "v_proj"]

Configuração de Treinamento:
- num_epochs: 3
- batch_size: 4
- learning_rate: 2e-4
- warmup_steps: 100
- max_grad_norm: 1.0
```

**6.3 Estratégia de Treinamento**
```
Fase 1: Preparação (semana 1)
- Coleta de dados
- Análise exploratória
- Pré-processamento

Fase 2: Desenvolvimento (semanas 2-4)
- Implementação backend
- Fine-tuning
- Primeiros testes

Fase 3: Integração (semanas 5-6)
- Interface web
- Testes fim-a-fim
- Otimizações

Fase 4: Avaliação (semana 7)
- Métricas finais
- Documentação
- Apresentação
```

#### Seção 7: Arquitetura do Sistema (1-2 páginas)

Incluir diagrama (ASCII art ou imagem):
```
┌─────────────────────┐
│   Interface Web     │
│  (HTML5/JavaScript) │
└──────────┬──────────┘
           │ HTTP
┌──────────▼──────────┐
│  FastAPI Server     │
│  (Backend Python)   │
└──────────┬──────────┘
           │ Memory
┌──────────▼──────────┐
│ LLaMA 2 + LoRA      │
│ (Modelo de IA)      │
└─────────────────────┘
```

Descrever fluxo de dados completo.

#### Seção 8: Métricas e Avaliação (1 página)

Como avaliará o sucesso:
```
1. Métrica de Qualidade
   - BLEU Score
   - ROUGE Score
   - Acurácia em teste set
   
2. Métrica de Performance
   - Tempo médio de resposta
   - Tempo de treinamento
   - Uso de memória
   
3. Métrica de Complexidade
   - Análise assintótica O(n)
   - Análise empírica com gráficos
   - Comparação esperado vs real
```

#### Seção 9: Cronograma (0.5 página)

Tabela com:
- Semana
- Atividades
- Responsável
- Status

#### Seção 10: Referências (0.5 página)

- Artigos científicos consultados
- Documentação oficial
- Repositórios GitHub
- Tutoriais seguidos

**Total esperado: 12-15 páginas de conteúdo técnico**

---

## FASE 2: IMPLEMENTAÇÃO ✓

### Código Entregável

Estrutura de arquivos esperada:

```
PAA_SeuNome_Matricula.zip
├── README.md (instruções para rodar)
├── requirements.txt (dependências Python)
├── Dataset/
│   ├── dados_brutos/ (dados originais)
│   ├── dados_processados.json (após limpeza)
│   └── dataset_analise.py (script de análise)
├── Coleta/
│   ├── collect_data.py
│   └── dados.json
├── Preprocessamento/
│   ├── preprocess.py
│   ├── tokenizer_utils.py
│   └── validation.py
├── Treinamento/
│   ├── fine_tuning.py
│   ├── model_config.py
│   ├── trainer.py
│   └── lora_config.json
├── Backend/
│   ├── main.py (FastAPI server)
│   ├── endpoints.py
│   ├── config.py
│   └── utils.py
├── Frontend/
│   ├── index.html
│   ├── styles.css (opcional)
│   └── script.js
├── Análise/
│   ├── complexidade_analysis.py
│   ├── metricas.py
│   └── graficos.py
└── Documentacao/
    ├── ANALISE_COMPLEXIDADE.md
    ├── GUIA_USUARIO.md
    ├── DECISOES_TECNICAS.md
    └── graficos/ (PNG/PDF dos gráficos)

```

### Código Mínimo Funcional

- [ ] Script de coleta de dados (`collect_data.py`)
- [ ] Script de pré-processamento (`preprocess.py`)
- [ ] Script de fine-tuning (`fine_tuning.py`)
- [ ] Backend FastAPI (`main.py`)
- [ ] Frontend HTML (`index.html`)
- [ ] Script de análise (`complexidade_analysis.py`)

### Testes e Validação

- [ ] Sistema roda sem erros
- [ ] Backend inicia na porta 8000
- [ ] Frontend carrega sem erros
- [ ] Pode fazer pergunta e receber resposta
- [ ] Interface é usável e responsiva

---

## FASE 3: ANÁLISE DE COMPLEXIDADE ✓

Este é o DIFERENCIAL para PAA!

### O que Deve Conter

- [ ] **Análise Teórica (Big-O)**
  - Tokenização: O(n × log k)
  - Self-Attention: O(n²)
  - Fine-tuning: O(E × N × r × d)
  - Inferência: O(m × n² × d)

- [ ] **Análise Empírica**
  - Medir tempo real de cada operação
  - Tabela com n vs tempo
  - Gráficos: linear, quadrático, exponencial
  - Comparar teórico vs prático

- [ ] **Comparação: Com vs Sem LoRA**
  - Tabela lado-a-lado
  - Memória: 84GB vs 4GB
  - Tempo: 450min vs 45min
  - Speedup: 10x

- [ ] **Gráficos Obrigatórios**
  1. Loss durante treinamento
  2. Tempo de resposta vs tamanho de entrada
  3. Memória utilizada
  4. Comparação complexidade teórica

### Formato de Análise

Arquivo: `ANALISE_COMPLEXIDADE.md`

```markdown
# Análise de Complexidade

## 1. Tokenização

### Análise Teórica
Complexidade: O(n × log k)
Justificativa: ...

### Análise Empírica
| n (tokens) | Tempo (ms) |
|-----------|-----------|
| 100       | 2.5       |
| 500       | 8.3       |
| 1000      | 15.2      |
| 2000      | 28.5      |

### Gráfico
[incluir gráfico]

### Conclusão
Comportamento observado: Linear
Matches teórico: Sim ✓
```

---

## FASE 4: APRESENTAÇÃO ✓

### Slides (20 minutos)

Estrutura recomendada:

1. **Capa** (1 slide)
   - Título do projeto
   - Nomes dos integrantes
   - Data

2. **Contexto** (2 slides)
   - O que é saúde pública?
   - Por que LLM para este tema?
   - Demonstração rápida

3. **Dados** (2 slides)
   - Fonte dos dados
   - Quantidade e distribuição
   - Exemplos de perguntas/respostas

4. **Fundamentação Teórica** (3 slides)
   - O que é um Transformer?
   - Self-attention explicado
   - Arquitetura geral (diagrama)

5. **Implementação** (2 slides)
   - Stack tecnológico (tabela)
   - Arquitetura do sistema (diagrama)

6. **Fine-tuning e LoRA** (2 slides)
   - Como funciona LoRA?
   - Comparação: com vs sem LoRA
   - Gráfico de redução de complexidade

7. **Análise de Complexidade** (3 slides)
   - Análise teórica de cada componente
   - Gráficos de performance empírica
   - Conclusões

8. **Resultados** (2 slides)
   - Métricas de qualidade
   - Tempo de resposta
   - Exemplos de perguntas/respostas bem-sucedidas

9. **Demonstração** (demo ao vivo - não em slide)
   - 4-5 perguntas diferentes
   - Mostrar tempo de processamento

10. **Conclusão** (1 slide)
    - Aprendizados principais
    - Limitações encontradas
    - Melhorias futuras

### Demonstração Ao Vivo

**Setup:**
```bash
# Terminal 1: Backend
source venv/bin/activate
python main.py

# Terminal 2: Frontend
# Abrir arquivo index.html no navegador
```

**Perguntas para Demonstrar:**
1. "Quais são os sintomas da dengue?"
2. "Como prevenir COVID-19?"
3. "O que é vacinação?"
4. "Qual a diferença entre vírus e bactéria?"
5. "Como funciona o sistema imunológico?"

**Métricas a Mostrar:**
- Tempo de resposta de cada pergunta
- Qualidade das respostas
- Interface responsiva

---

## FASE 5: ENTREGA FINAL ✓

### Arquivo a Enviar

**Nome:** `PAA_PrimeiroNome_Matricula_Proj.zip`

**Conteúdo obrigatório:**
- [ ] Documento PDF com seções 1-10 (técnico)
- [ ] Código-fonte comentado
- [ ] README.md com instruções
- [ ] requirements.txt
- [ ] Dataset (ou script para baixar)
- [ ] Análise de complexidade (gráficos)
- [ ] Slides da apresentação (PDF)
- [ ] Arquivo de configuração (se houver)

**Conteúdo opcional mas valioso:**
- [ ] Testes unitários
- [ ] Dockerfile
- [ ] Notebook Jupyter com análise
- [ ] Vídeo de demonstração

### Checklist Final

#### Documentação
- [ ] PDF técnico completo (12-15 páginas)
- [ ] Todas as seções preenchidas
- [ ] Análise de complexidade rigorosa
- [ ] Referências adequadas
- [ ] Sem erros de português/formato

#### Código
- [ ] Sem erros de sintaxe
- [ ] Bem comentado
- [ ] Segue boas práticas Python
- [ ] Modular e reutilizável
- [ ] Todos os requisitos instaláveis

#### Funcionalidade
- [ ] Sistema roda sem erros
- [ ] Pode fazer perguntas e receber respostas
- [ ] Interface é amigável
- [ ] Tempo de resposta aceitável

#### Apresentação
- [ ] Slides claros e profissionais
- [ ] Demonstração testada e funcionando
- [ ] Todos entendem cada parte
- [ ] Tempo de apresentação respeitado (20 min)
- [ ] Análise de complexidade explicada bem

#### Requisitos PAA
- [ ] Análise de complexidade O(n) teórica
- [ ] Validação empírica com dados reais
- [ ] Gráficos comparativos
- [ ] Justificação de decisões algorítmicas
- [ ] Discussão de trade-offs

---

## SCORING ESPERADO

Para obter nota máxima:

| Critério | Peso | Como Garantir |
|----------|------|---------------|
| Documentação Técnica | 20% | Completa, detalhada, sem erros |
| Código Funcional | 20% | Sem bugs, bem estruturado |
| Análise de Complexidade | 25% | Teórica + empírica + gráficos |
| Apresentação | 20% | Clara, profissional, demonstração |
| Inovação/Qualidade | 15% | Extras: otimizações, testes, etc |

---

## DICAS FINAIS

### ✅ Faça
1. Comece cedo (não deixe para última semana)
2. Documente tudo enquanto escreve código
3. Teste frequentemente
4. Mantenha análise atualizada
5. Prepare apresentação com antecedência
6. Teste demo ao vivo múltiplas vezes

### ❌ Evite
1. Código desorganizado/sem comentários
2. Análise de complexidade superficial
3. Deixar tudo para última noite
4. Demo sem testes prévios
5. Apresentação apressada
6. Tecnologias não open-source

---

## CONTATO E SUPORTE

Se tiver dúvidas:

1. **Sobre o projeto:** Revise este documento
2. **Sobre código:** Consulte `guia_codigo_LLM.md`
3. **Sobre teoria:** Consulte `pesquisa_LLM_saude.md`
4. **Sobre arquitetura:** Consulte `arquitetura_diagramas.md`

---

## FÓRMULA DO SUCESSO

```
SUCESSO = Planejamento (20%)
        + Implementação (20%)
        + Análise técnica (25%)
        + Apresentação (20%)
        + Qualidade (15%)

Não pule nenhum desses passos!
```

---

**Boa sorte! Vocês conseguem! 🚀**

Data de Entrega: **5 de novembro de 2025 - 23:55**

