# RESUMO EXECUTIVO: Projeto LLM para Saúde Pública

**Data:** Novembro 2025  
**Disciplina:** Projeto e Análise de Algoritmos (PAA)  
**Universidade:** UnB - Departamento de Ciência da Computação  

---

## 📋 VISÃO GERAL

Seu grupo desenvolvará um **sistema de Inteligência Artificial (LLM)** capaz de responder perguntas em linguagem natural sobre **Saúde Pública**.

**O sistema funcionará assim:**

```
Usuário pergunta:
"O que é dengue?"
        ↓
[Servidor Backend processa com IA]
        ↓
Sistema responde:
"Dengue é uma doença infecciosa..."
        ↓
Resposta exibida em interface web
```

---

## 🎯 OBJETIVOS PRINCIPAIS

1. **Coletar dados** sobre saúde pública
2. **Treinar modelo LLM** com esses dados
3. **Criar interface web** para interação
4. **Analisar complexidade** dos algoritmos
5. **Apresentar resultados** à turma

---

## 🛠️ TECNOLOGIA RECOMENDADA

| Componente | Tecnologia | Por Quê? |
|-----------|-----------|---------|
| **Linguagem** | Python | Melhor ecossistema para ML/NLP |
| **Framework ML** | PyTorch | Flexível, moderno, padrão ouro |
| **Modelo Base** | LLaMA 2 7B | Open-source, bom custo/benefício |
| **Fine-tuning** | LoRA | 95% menos memória, 10x mais rápido |
| **Backend** | FastAPI | Rápido, fácil, Pythônico |
| **Frontend** | HTML5 + JavaScript | Simples, não requer build tools |

**Todas as tecnologias são OPEN SOURCE com licenças apropriadas** ✅

---

## 📊 STACK FINAL

```
┌─────────────────────────────────────┐
│   INTERFACE WEB                      │
│   (HTML5 + JavaScript)               │
└────────────┬────────────────────────┘
             │ HTTP
┌────────────▼────────────────────────┐
│   FASTAPI (Python)                   │
│   - API REST                         │
│   - Processa perguntas              │
│   - Retorna respostas               │
└────────────┬────────────────────────┘
             │ GPU Memory
┌────────────▼────────────────────────┐
│   LLAMA 2 7B + LoRA                 │
│   - Modelo base pré-treinado        │
│   - Adaptadores especializados      │
│   - Treinado em dados de SP         │
└──────────────────────────────────────┘
```

---

## 💾 DADOS

### Fontes Recomendadas:
- **WHO** (Organização Mundial de Saúde)
- **MedQA Dataset** (modificado)
- **PubMed Central** (artigos científicos)
- **Seu próprio dataset** (manualmente coletado)

### Requisitos:
- ✅ Mínimo 500 exemplos
- ✅ Máximo 10.000 exemplos
- ✅ Balanceado por categoria
- ✅ Revisado por qualidade

### Estrutura:
```json
{
  "pergunta": "O que é dengue?",
  "resposta": "Dengue é uma doença infecciosa...",
  "categoria": "doenças_infecciosas",
  "confiança": 0.95
}
```

---

## 🧠 COMO FUNCIONA O TREINAMENTO

### Passo 1: Modelo Pré-Treinado
```
LLaMA 2 7B (fornecido pela Meta)
↓
Já conhece padrões gerais de linguagem
↓
Mas não sabe sobre saúde pública especificamente
```

### Passo 2: Fine-Tuning com LoRA
```
Seus dados de saúde pública
↓
Passados pelo modelo durante treinamento
↓
Modelo aprende a responder sobre seu tema
↓
Apenas 0.2% dos parâmetros são atualizados (LoRA)
```

### Passo 3: Resultado
```
Modelo especializado em saúde pública
↓
Pronto para responder perguntas
↓
Em tempo real via interface web
```

---

## ⚡ ANÁLISE DE COMPLEXIDADE (PARA SEU RELATÓRIO)

### Tokenização
```
Complexidade: O(n × log k)
n = caracteres, k = vocab size
Espaço: O(v) = 50.000 tokens
```

### Self-Attention (Transformer)
```
Complexidade: O(n²)
n = comprimento da sequência

Exemplo:
- 100 tokens → 10.000 operações
- 1.000 tokens → 1.000.000 operações
- 2.000 tokens → 4.000.000 operações
```

### Fine-Tuning
```
SEM LoRA:
Parâmetros: 7 bilhões
Memória: 28 GB

COM LoRA:
Parâmetros: 13 milhões (0.2%)
Memória: 262 KB
Speedup: 10x mais rápido
```

### Inferência (Responder Pergunta)
```
Complexidade: O(n × d²)
n = tokens gerados
d = dimensão do modelo

Tempo típico: 3-5 segundos por resposta
Em GPU moderna: ~2-3 segundos
```

---

## 📅 CRONOGRAMA SUGERIDO

| Semana | Atividades | Horas |
|--------|-----------|-------|
| 1 | Setup, requisitos, dataset | 15h |
| 2 | Análise de dados, pré-processamento | 12h |
| 3 | Implementação backend | 20h |
| 4 | Fine-tuning do modelo | 15h |
| 5 | Frontend + integração | 10h |
| 6 | Testes, documentação, análise | 12h |
| 7 | Preparação apresentação | 10h |
| **Total** | | **94h** |

**Por pessoa:** 94h ÷ n_pessoas

---

## 🎬 O QUE APRESENTAR À TURMA

### 1. Slides (20 minutos)
- ✅ Explicar o tema e por que usou IA
- ✅ Mostrar arquitetura do sistema
- ✅ Explicar Transformers (visualmente)
- ✅ Análise de complexidade com gráficos
- ✅ Resultados e métricas

### 2. Demonstração Ao Vivo (10 minutos)
- ✅ Interface web funcionando
- ✅ 4-5 perguntas diferentes
- ✅ Mostrar tempo de resposta
- ✅ Exemplos de casos de sucesso

### 3. Discussão Técnica (5 minutos)
- ✅ Desafios encontrados
- ✅ Soluções implementadas
- ✅ Limitações do sistema

---

## 💡 DICAS PARA O SUCESSO

### ✅ Faça:
1. **Use LoRA** - Reduz complexidade drasticamente
2. **Comece pequeno** - 500 exemplos é suficiente para demo
3. **Documente tudo** - Cada decisão deve ser justificada
4. **Teste frequentemente** - Não espere terminar tudo
5. **Rastreie métricas** - Mantenha gráficos do progresso

### ❌ Evite:
1. ❌ Treinar modelo completo (28GB memória necessária)
2. ❌ Usar dados não verificados (qualidade importante)
3. ❌ Descuidar da análise de complexidade (requisito PAA)
4. ❌ Deixar tudo para última semana
5. ❌ Tentar GPT-4 (não é open-source, violaria requisito)

---

## 🚀 PRÓXIMOS PASSOS

### Imediato (esta semana):
1. ✅ Ler este documento completamente
2. ✅ Revisar arquivo `pesquisa_LLM_saude.md` com detalhes teóricos
3. ✅ Revisar arquivo `guia_codigo_LLM.md` com exemplos práticos
4. ✅ Reunir-se com seu grupo
5. ✅ Começar setup do ambiente Python

### Curto prazo (próximas 2 semanas):
1. Definir e coletar dataset
2. Implementar scripts de pré-processamento
3. Familiarizar-se com PyTorch e Transformers
4. Fazer primeiro treinamento teste

### Médio prazo (semanas 3-5):
1. Refinar dataset e modelo
2. Desenvolver backend FastAPI
3. Criar interface web
4. Testes e ajustes

### Longo prazo (semanas 6-7):
1. Otimizações finais
2. Documentação completa
3. Preparação de apresentação
4. Análise de complexidade final

---

## 📚 ARQUIVOS DE SUPORTE

Você recebeu **3 documentos**:

1. **pesquisa_LLM_saude.md** (15 páginas)
   - Fundamentação teórica completa
   - Explicação de Transformers e LLMs
   - Metodologias de fine-tuning
   - Análise detalhada de complexidade
   - Referências bibliográficas

2. **guia_codigo_LLM.md** (10 páginas)
   - Código Python pronto para usar
   - Scripts de coleta de dados
   - Implementação de fine-tuning
   - Backend FastAPI completo
   - Interface web HTML/JS

3. **resumo_executivo.md** (este arquivo)
   - Visão geral do projeto
   - Guia rápido de referência
   - Cronograma e próximos passos

---

## 🎓 VALOR EDUCACIONAL

Este projeto ensina:

- 🧠 **Processamento de Linguagem Natural** (NLP)
- 🔬 **Deep Learning** com Transformers
- ⚡ **Otimização de Algoritmos** (LoRA, quantização)
- 🏗️ **Arquitetura de Sistemas** (backend, frontend)
- 📊 **Análise de Complexidade** (O-grande, empiricamente)
- 🔧 **Engenharia de ML** (dados, treino, deployment)
- 💻 **Full-stack Development** (Python, JavaScript, Web)

---

## 🤝 DISTRIBUIÇÃO DE TRABALHO (Sugestão)

Para grupo de **4 pessoas**:

**Pessoa 1 - Data & Backend:**
- Coleta e pré-processamento de dados
- Implementação FastAPI

**Pessoa 2 - ML & Training:**
- Setup de ambiente ML
- Fine-tuning do modelo
- Análise de complexidade

**Pessoa 3 - Frontend & UI:**
- Interface HTML/JavaScript
- Integração com backend
- Testes de usabilidade

**Pessoa 4 - Documentação & Apresentação:**
- Escrever relatório técnico
- Criar slides
- Preparar demonstração

---

## ❓ PERGUNTAS FREQUENTES

**P: Preciso de GPU?**  
R: LoRA funciona em GPUs com 6-8GB (você provavelmente tem isso)

**P: Posso usar meu laptop?**  
R: Sim, mas vai ser mais lento. GPU recomendada, CPU funciona também

**P: E se não tiver GPU?**  
R: Use Google Colab (grátis, com GPU T4)

**P: Quanto tempo de treinamento?**  
R: ~45 min em V100, ~2h em GPU modesta, ~6h em CPU

**P: Preciso de muitos dados?**  
R: Não! 500 exemplos já é suficiente para demo

**P: Como garanto qualidade das respostas?**  
R: Dataset de qualidade + ajuste de temperatura + validação

---

## 📞 SUPORTE TÉCNICO

Se encontrar problemas:

1. **Erro ao instalar PyTorch:** Visite pytorch.org para instruções GPU/CPU específicas
2. **Modelo não carrega:** Verificar espaço em disco (7B = ~13GB)
3. **FastAPI não funciona:** `pip install python-dotenv` e verificar porta 8000
4. **Resposta muito genérica:** Aumentar epochs ou melhorar dataset

---

## 🏆 OBJETIVO FINAL

Ao terminar este projeto, você terá:

✅ Um **sistema de IA funcionando**  
✅ **Apresentação técnica** impressionante  
✅ **Compreensão profunda** de LLMs e NLP  
✅ **Código bem documentado** e reutilizável  
✅ **Análise rigorosa** de complexidade algorítmica  
✅ **Prática em full-stack development**  
✅ **Portfólio impressionante** para carreira  

---

## 📝 CHECKLIST FINAL

Antes de apresentar, verifique:

- [ ] Código está limpo e comentado
- [ ] Análise de complexidade está correta e justificada
- [ ] Sistema funciona sem erros (demo testada)
- [ ] Dados são de fonte confiável
- [ ] Relatório técnico é completo
- [ ] Slides são claros e visualmente interessantes
- [ ] Todos os integrantes entendem cada parte
- [ ] Licenças open-source estão verificadas
- [ ] Cronograma foi respeitado
- [ ] Qualidade de apresentação é profissional

---

**Boa sorte! Vocês conseguem! 🚀**

*Qualquer dúvida específica, consulte os documentos de pesquisa e código fornecidos.*

