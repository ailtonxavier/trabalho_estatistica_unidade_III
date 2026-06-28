import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
import seaborn as sns
import os

# 1. PREPARAÇÃO DO AMBIENTE
if not os.path.exists('images'): os.makedirs('images')

# 2. CARREGAMENTO DOS DADOS (Nome atualizado)
df = pd.read_csv('database/dados_estudantes.csv')

# 3. LIMPEZA DOS DADOS (Etapa 2)
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        df[col] = df[col].fillna(df[col].mean())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

# Dicionário de tradução para gráficos e matrizes
traducao = {
    'Pre_Semester_GPA': 'GPA Pré',
    'Weekly_GenAI_Hours': 'Horas IA',
    'Traditional_Study_Hours': 'Horas Estudo',
    'Post_Semester_GPA': 'GPA Final',
    'Burnout_Risk_Level': 'Risco Burnout'
}

# 4. ANÁLISE EXPLORATÓRIA (Etapa 3 - Gráficos)
quantitativas = ['Pre_Semester_GPA', 'Weekly_GenAI_Hours', 'Traditional_Study_Hours', 'Post_Semester_GPA']
df_plot = df.rename(columns=traducao)

for col in [traducao[c] for c in quantitativas]:
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(df_plot[col], kde=True, color='blue')
    plt.title(f'Distribuição: {col}')
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df_plot[col], color='orange')
    plt.title(f'Boxplot: {col}')
    plt.tight_layout()
    plt.savefig(f'images/estatistica_{col.replace(" ", "_")}.png')
    plt.close()

# 5. MATRIZ DE CORRELAÇÃO (Etapa 4)
df_matriz = df.select_dtypes(include=['float64', 'int64']).rename(columns=traducao)
plt.figure(figsize=(10, 8))
sns.heatmap(df_matriz.corr(), annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 8})
plt.title('Matriz de Correlação')
plt.tight_layout()
plt.savefig('images/matriz_correlacao.png')
plt.close()

# 6. ANOVA E GRÁFICO (Etapa 6)
plt.figure(figsize=(8, 5))
sns.boxplot(x='Burnout_Risk_Level', y='Post_Semester_GPA', data=df)
plt.title('GPA Final por Nível de Risco de Burnout')
plt.savefig('images/anova_boxplot.png')
plt.close()

# 7. REGRESSÃO LINEAR MÚLTIPLA (Etapa 7)
y = df['Post_Semester_GPA']
X = sm.add_constant(df[['Pre_Semester_GPA', 'Weekly_GenAI_Hours', 'Traditional_Study_Hours']])
modelo = sm.OLS(y, X).fit()

# EXIBIÇÃO DOS RESULTADOS NO TERMINAL
print("="*60)
print(" RESULTADOS ESTATÍSTICOS (Para copiar para o seu Relatório)")
print("="*60)
print(modelo.summary())

# Teste de Normalidade dos Resíduos
jb_stat, jb_pval = stats.jarque_bera