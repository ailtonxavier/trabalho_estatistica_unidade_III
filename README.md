# Relatório de Estatística Aplicada - III Unidade (2026.1)

Este repositório contém a análise estatística realizada sobre a base de dados **"AI Impact on Students"**, conforme solicitado na atividade da III Unidade da disciplina de Estatística Aplicada (TAD0022).

## 1. Domínio de Aplicação
* **Cenário:** Estudo sobre o impacto do uso de ferramentas de Inteligência Artificial Generativa (GenAI) no desempenho acadêmico, saúde mental (Burnout) e retenção de habilidades dos estudantes.
* **Propósito:** Analisar correlações entre horas de uso de IA, horas de estudo tradicional e o GPA (média acadêmica) final dos alunos.
* **Referência:** [Kaggle - AI Impact on Students Dataset](https://www.kaggle.com/datasets/laveshjadon/ai-impact-on-students)

## 2. Estrutura do Projeto
* `db.csv`: Base de dados original contendo os registros dos estudantes.
* `main.py`: Script principal que realiza a limpeza, processamento, ANOVA e Regressão Linear.
* `requirements.txt`: Dependências necessárias para executar o ambiente.
* `distribuicao_gpa.png`: Gráfico gerado automaticamente com a distribuição do GPA Final.

## 3. Metodologia
O pipeline de análise foi dividido conforme as exigências da disciplina:
1. **Limpeza de Dados:** Tratamento de valores nulos substituindo-os pela média (variáveis quantitativas) ou moda (variáveis qualitativas).
2. **Análise de Variância (ANOVA):** Comparação das médias de GPA Final entre diferentes níveis de risco de Burnout, seguida por teste de Tukey para identificação de diferenças entre grupos.
3. **Regressão Linear Múltipla:** Ajuste de modelo para explicar o `Post_Semester_GPA` em função do `Pre_Semester_GPA`, `Weekly_GenAI_Hours` e `Traditional_Study_Hours`.

## 4. Como Executar
Para rodar este projeto em sua máquina:

1. Certifique-se de ter o Python 3.12 instalado.
2. Crie um ambiente virtual e instale as dependências:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # No Windows use: venv\Scripts\activate
   pip install -r requirements.txt