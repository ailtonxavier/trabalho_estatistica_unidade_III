import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
import seaborn as sns
import os
from statsmodels.stats.diagnostic import het_breuschpagan

# 1. PREPARAÇÃO DO AMBIENTE
if not os.path.exists('images'): os.makedirs('images')
if not os.path.exists('results'): os.makedirs('results')

# 2. CARREGAMENTO DOS DADOS
df = pd.read_csv('database/dados_estudantes.csv')

# 3. LIMPEZA DOS DADOS (Etapa 2)
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        df[col] = df[col].fillna(df[col].mean())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

traducao = {
    'Pre_Semester_GPA': 'GPA Pré',
    'Weekly_GenAI_Hours': 'Horas IA',
    'Traditional_Study_Hours': 'Horas Estudo',
    'Post_Semester_GPA': 'GPA Final',
    'Burnout_Risk_Level': 'Risco Burnout'
}

# 4. ANÁLISE EXPLORATÓRIA (Etapa 3)
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
traducao_burnout = {'Low': 'Baixo', 'Medium': 'Médio', 'High': 'Alto'}
df['Risco_Burnout_PT'] = df['Burnout_Risk_Level'].map(traducao_burnout)
df['Risco_Burnout_PT'] = pd.Categorical(df['Risco_Burnout_PT'], categories=['Baixo', 'Médio', 'Alto'], ordered=True)
plt.figure(figsize=(8, 5))
sns.boxplot(x='Risco_Burnout_PT', y='Post_Semester_GPA', data=df)
plt.title('GPA Final por Nível de Risco de Burnout')
plt.xlabel('Nível de Risco')
plt.ylabel('GPA Final')
plt.savefig('images/anova_boxplot.png')
plt.close()

# 7. REGRESSÃO LINEAR MÚLTIPLA (Etapa 7)
y = df['Post_Semester_GPA']
X = sm.add_constant(df[['Pre_Semester_GPA', 'Weekly_GenAI_Hours', 'Traditional_Study_Hours']])
modelo = sm.OLS(y, X).fit()

# Salvar resultados em TXT na pasta results
with open('results/resultado_regressao.txt', 'w') as f:
    f.write(str(modelo.summary()))
    f.write("\n\n--- Testes de Diagnóstico ---\n")
    jb_stat, jb_pval = stats.jarque_bera(modelo.resid)
    f.write(f"Normalidade (Jarque-Bera) p-valor: {jb_pval:.4e}\n")
    bp_test = het_breuschpagan(modelo.resid, modelo.model.exog)
    f.write(f"Homocedasticidade (Breusch-Pagan) p-valor: {bp_test[1]:.4e}\n")

print("Processamento concluído. Verifique as pastas 'images/' e 'results/'.")