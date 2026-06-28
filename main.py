import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
import seaborn as sns
import os

# Cria a pasta de imagens se ela não existir
if not os.path.exists('images'):
    os.makedirs('images')

# 1. CARREGAMENTO DOS DADOS (Caminho atualizado)
df = pd.read_csv('database/db.csv')

# --- ETAPA 2: LIMPEZA ---
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        df[col] = df[col].fillna(df[col].mean())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

# Dicionário de tradução fixo
traducao_colunas = {
    'Student_ID': 'ID Aluno',
    'Pre_Semester_GPA': 'GPA Pré-Semestre',
    'Weekly_GenAI_Hours': 'Horas IA Semanal',
    'Traditional_Study_Hours': 'Horas Estudo Tradicional',
    'Post_Semester_GPA': 'GPA Final',
    'Anxiety_Level_During_Exams': 'Nível Ansiedade Provas',
    'Skill_Retention_Score': 'Nota Retenção Habilidade',
    'Tool_Diversity': 'Diversidade Ferramentas'
}

# --- ETAPA 3: ANÁLISE EXPLORATÓRIA ---
# Histograma
plt.figure(figsize=(6, 4))
sns.histplot(df['Post_Semester_GPA'], kde=True, color='skyblue')
plt.title('Distribuição GPA Final')
plt.xlabel('GPA Final')
plt.ylabel('Frequência')
plt.tight_layout()
plt.savefig('images/distribuicao_gpa.png') # Pasta images
plt.close()

# Boxplot
plt.figure(figsize=(6, 4))
sns.boxplot(y=df['Post_Semester_GPA'], color='lightgreen')
plt.title('Boxplot GPA Final')
plt.ylabel('GPA Final')
plt.tight_layout()
plt.savefig('images/boxplot_gpa.png') # Pasta images
plt.close()

# --- ETAPA 4: MATRIZ DE CORRELAÇÃO ---
df_plot = df.rename(columns=traducao_colunas)
colunas_pt = [traducao_colunas[c] for c in traducao_colunas if c in df.columns]
correlacao = df_plot[colunas_pt].corr(method='pearson')

plt.figure(figsize=(10, 8))
sns.heatmap(correlacao, annot=True, cmap='coolwarm', fmt=".2f", 
            xticklabels=correlacao.columns, yticklabels=correlacao.columns)
plt.title('Matriz de Correlação')
plt.tight_layout()
plt.savefig('images/matriz_correlacao.png') # Pasta images
plt.close()

print("Processamento concluído. Imagens salvas na pasta 'images/'.")