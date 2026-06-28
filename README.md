# Relatório de Estatística Aplicada - III Unidade (2026.1)

Este repositório contém a análise estatística realizada sobre a base de dados **"AI Impact on Students"**, conforme solicitado na atividade da III Unidade da disciplina de Estatística Aplicada (TAD0022).

## 1. Domínio de Aplicação
* **Cenário:** Estudo sobre o impacto do uso de ferramentas de Inteligência Artificial Generativa (GenAI) no desempenho acadêmico, saúde mental (Burnout) e retenção de habilidades dos estudantes.
* **Propósito:** Analisar correlações entre horas de uso de IA, horas de estudo tradicional e o GPA (média acadêmica) final dos alunos.
* **Referência:** [Kaggle - AI Impact on Students Dataset](https://www.kaggle.com/datasets/laveshjadon/ai-impact-on-students)

## 2. Estrutura do Projeto
* `database/`: Contém o arquivo `dados_estudantes.csv`.
* `images/`: Contém os gráficos gerados automaticamente (`.png`):
    * Distribuições e Boxplots individuais;
    * Matriz de correlação;
    * Boxplot da ANOVA.
* `results/`: Contém o arquivo `resultado_regressao.txt` com o resumo estatístico do modelo.
* `main.py`: Script principal de processamento e análise.
* `requirements.txt`: Dependências do projeto.

## 3. Metodologia
O pipeline de análise foi estruturado nas seguintes etapas:
1. **Limpeza de Dados:** Tratamento de valores nulos utilizando média (quantitativas) e moda (qualitativas).
2. **Análise Exploratória:** Geração de estatísticas descritivas visuais (histogramas e boxplots) e mapa de calor de correlação.
3. **Análise de Variância (ANOVA):** Comparação das médias de GPA entre níveis de Risco de Burnout, com representação visual.
4. **Regressão Linear Múltipla:** Modelagem do `GPA Final` em função de `GPA Pré-Semestre`, `Horas IA` e `Horas de Estudo Tradicional`, incluindo diagnósticos de normalidade (Jarque-Bera) e homocedasticidade (Breusch-Pagan).

## 4. Como Executar
1. Instale as dependências:
   ```bash
   pip install -r requirements.txt