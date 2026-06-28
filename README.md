# Relatório de Estatística Aplicada - III Unidade (2026.1)

Este repositório contém a análise estatística realizada sobre a base de dados **"AI Impact on Students"**, conforme solicitado na atividade da III Unidade da disciplina de Estatística Aplicada (TAD0022).

## 1. Domínio de Aplicação
* **Cenário:** Estudo sobre o impacto do uso de ferramentas de Inteligência Artificial Generativa (GenAI) no desempenho acadêmico, saúde mental (Burnout) e retenção de habilidades dos estudantes.
* **Propósito:** Analisar correlações entre horas de uso de IA, horas de estudo tradicional e o GPA (média acadêmica) final dos alunos.
* **Referência:** [Kaggle - AI Impact on Students Dataset](https://www.kaggle.com/datasets/laveshjadon/ai-impact-on-students)

## 2. Estrutura do Projeto
* `db.csv`: Base de dados original contendo os registros dos estudantes.
* `main.py`: Script principal que realiza a limpeza, processamento, ANOVA, Regressão Linear e geração de gráficos.
* `requirements.txt`: Dependências necessárias para executar o ambiente.
* `distribuicao_gpa.png`: Gráfico de distribuição do GPA Final.
* `boxplot_gpa.png`: Gráfico de caixa para identificação de outliers do GPA Final.
* `matriz_correlação.png`: Mapa de calor da correlação entre as variáveis quantitativas.

## 3. Metodologia
O pipeline de análise foi dividido conforme as exigências da disciplina:
1. **Limpeza de Dados:** Tratamento de valores nulos substituindo-os pela média (variáveis quantitativas) ou moda (variáveis qualitativas).
2. **Análise Exploratória:** Geração de estatísticas descritivas e visualização através de histogramas, boxplots e matriz de correlação (com tradução de termos para o português).
3. **Análise de Variância (ANOVA):** Comparação das médias de GPA Final entre diferentes níveis de risco de Burnout, seguida por teste de Tukey (se aplicável).
4. **Regressão Linear Múltipla:** Ajuste de modelo para explicar o `GPA Final` em função do `GPA Pré-Semestre`, `Horas IA Semanal` e `Horas Estudo Tradicional`, incluindo testes de normalidade (Jarque-Bera) e homocedasticidade (Breusch-Pagan).

## 4. Como Executar
Para rodar este projeto em sua máquina:

1. Certifique-se de ter o Python 3.12+ instalado.
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt