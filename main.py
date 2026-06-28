import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.diagnostic import het_breuschpagan
import seaborn as sns

# 1. CARREGAMENTO DOS DADOS
df = pd.read_csv('db.csv')

# --- ETAPA 2: CRÍTICA E LIMPEZA DOS DADOS ---
print("="*60)
print(" ETAPA 2 - CRÍTICA E LIMPEZA DOS DADOS")
print("="*60)
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        df[col] = df[col].fillna(df[col].mean())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])
print("Limpeza concluída.\n")

# --- ETAPA 3: ANÁLISE EXPLORATÓRIA (Gerando Gráficos) ---
print("="*60)
print(" ETAPA 3 - ANÁLISE EXPLORATÓRIA")
print("="*60)

# Gráfico 1: Distribuição e Boxplot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['Post_Semester_GPA'], kde=True)
plt.title('Distribuição GPA Final')
plt.subplot(1, 2, 2)
sns.boxplot(y=df['Post_Semester_GPA'])
plt.title('Boxplot GPA Final')
plt.tight_layout()
plt.savefig('analise_exploratoria.png')
plt.close() # Fecha a figura após salvar
print("Gráfico 'analise_exploratoria.png' gerado.")

# --- ETAPA 4: MATRIZ DE CORRELAÇÃO ---
print("\n" + "="*60)
print(" ETAPA 4 - MATRIZ DE CORRELAÇÃO")
print("="*60)
colunas_num = df.select_dtypes(include=['float64', 'int64']).columns
correlacao = df[colunas_num].corr(method='pearson')

plt.figure(figsize=(10, 8))
sns.heatmap(correlacao, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlação')
plt.tight_layout()
plt.savefig('matriz_correlação.png')
plt.close() # Fecha a figura após salvar
print("Gráfico 'matriz_correlação.png' gerado.")

# --- ETAPA 5: TESTES DE HIPÓTESES ---
print("\n" + "="*60)
print(" ETAPA 5 - TESTES DE HIPÓTESES")
print("="*60)
grupo1 = df[df['Paid_Subscription'] == True]['Post_Semester_GPA']
grupo2 = df[df['Paid_Subscription'] == False]['Post_Semester_GPA']
t_stat, p_val = stats.ttest_ind(grupo1, grupo2)
print(f"Teste T (Paid_Subscription): Valor-p = {p_val:.4f}")

# --- ETAPA 6: ANOVA ---
print("\n" + "="*60)
print(" ETAPA 6 - ANOVA")
print("="*60)
grupos = [group['Post_Semester_GPA'] for name, group in df.groupby('Burnout_Risk_Level')]
f_stat, p_val_anova = stats.f_oneway(*grupos)
print(f"Resultado ANOVA: F={f_stat:.4f}, Valor-p={p_val_anova:.4e}")

# --- ETAPA 7: REGRESSÃO LINEAR MÚLTIPLA ---
print("\n" + "="*60)
print(" ETAPA 7 - REGRESSÃO LINEAR MÚLTIPLA")
print("="*60)
y = df['Post_Semester_GPA']
X = df[['Pre_Semester_GPA', 'Weekly_GenAI_Hours', 'Traditional_Study_Hours']]
X = sm.add_constant(X)
modelo = sm.OLS(y, X).fit()
print(modelo.summary())