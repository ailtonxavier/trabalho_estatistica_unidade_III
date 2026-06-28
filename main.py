import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.diagnostic import het_breuschpagan

# 1. CARREGAMENTO DOS DADOS
df = pd.read_csv('db.csv')

print("="*60)
print(" ETAPA 2 - CRÍTICA E LIMPEZA DOS DADOS")
print("="*60)
nulos_antes = df.isnull().sum().sum()
print(f"Total de valores nulos encontrados: {nulos_antes}")

# Tratamento: Média para quantitativas, Moda para qualitativas
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        df[col] = df[col].fillna(df[col].mean())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

print("Limpeza concluída! Dados formatados e valores faltantes preenchidos.\n")

# 2. ANOVA (Etapa 6)
print("="*60)
print(" ETAPA 6 - ANÁLISE DE VARIÂNCIA (ANOVA)")
print("="*60)
print("Objetivo: Verificar se o GPA Final (Post_Semester_GPA) varia conforme o Risco de Burnout.\n")
print("Hipótese Nula (H0): As médias do GPA são iguais para todos os níveis de Burnout.")
print("Hipótese Alternativa (H1): Pelo menos um nível de Burnout apresenta média de GPA diferente.\n")

grupos = [grupo['Post_Semester_GPA'].values for nome, grupo in df.groupby('Burnout_Risk_Level')]
f_stat, p_val = stats.f_oneway(*grupos)

print(f"Estatística F: {f_stat:.4f}")
print(f"Valor-p: {p_val:.4e}")

if p_val < 0.05:
    print("Conclusão: Rejeitamos H0. Existe diferença significativa entre os grupos.")
    print("Executando teste de Tukey para identificar quais grupos diferem...")
    tukey = pairwise_tukeyhsd(endog=df['Post_Semester_GPA'], groups=df['Burnout_Risk_Level'], alpha=0.05)
    print(tukey)
else:
    print("Conclusão: Não há evidência estatística para rejeitar H0 (as médias são semelhantes).")

# 3. REGRESSÃO (Etapa 7)
print("\n" + "="*60)
print(" ETAPA 7 - MODELO DE REGRESSÃO LINEAR MÚLTIPLA")
print("="*60)

y = df['Post_Semester_GPA']
X = df[['Pre_Semester_GPA', 'Weekly_GenAI_Hours', 'Traditional_Study_Hours']]
X = sm.add_constant(X)

modelo = sm.OLS(y, X).fit()

print("Equação obtida:")
print(f"GPA_Final = {modelo.params['const']:.4f} + ({modelo.params['Pre_Semester_GPA']:.4f} * Pre_GPA) + ({modelo.params['Weekly_GenAI_Hours']:.4f} * Horas_IA) + ({modelo.params['Traditional_Study_Hours']:.4f} * Horas_Estudo)")

print("\nQualidade do Modelo:")
print(f"Coeficiente de Determinação (R²): {modelo.rsquared:.4f}")
print(f"Significância Global (Valor-p do F): {modelo.f_pvalue:.4e}")

# Testes de Resíduos traduzidos
jb_stat, jb_pval = stats.jarque_bera(modelo.resid)
print(f"\nTeste de Normalidade dos Resíduos (Jarque-Bera): Valor-p = {jb_pval:.4e}")
print("-> Resíduos seguem distribuição normal" if jb_pval > 0.05 else "-> Resíduos não seguem distribuição normal")

bp_test = het_breuschpagan(modelo.resid, modelo.model.exog)
print(f"Teste de Homocedasticidade (Breusch-Pagan): Valor-p = {bp_test[1]:.4e}")
print("-> Variância constante (Homocedástico)" if bp_test[1] > 0.05 else "-> Variância não constante (Heterocedástico)")

# Gerando gráfico
plt.hist(df['Post_Semester_GPA'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribuição do GPA Final')
plt.xlabel('GPA')
plt.ylabel('Frequência')
plt.savefig('distribuicao_gpa.png')
print("\n[!] Gráfico salvo como 'distribuicao_gpa.png' para seu relatório.")