#!/usr/bin/env python
# coding: utf-8

# # Projeto Ciência de Dados - Previsão de Vendas
# 
# - Nosso desafio é conseguir prever as vendas que vamos ter em determinado período com base nos gastos em anúncios nas 3 grandes redes que a empresa Hashtag investe: TV, Jornal e Rádio
# 
# - Base de Dados: https://drive.google.com/drive/folders/1o2lpxoi9heyQV1hIlsHXWSfDkBPtze-V?usp=sharing

# # Projeto Ciência de Dados - Previsão de Vendas
# 
# - Nosso desafio é conseguir prever as vendas que vamos ter em determinado período com base nos gastos em anúncios nas 3 grandes redes que a empresa Hashtag investe: TV, Jornal e Rádio
# - TV, Jornal e Rádio estão em milhares de reais
# - Vendas estão em milhões

# #### Importar a Base de dados

# In[1]:


# Passo 3: Extração/Obtenção de Dados
import pandas as pd

tabela = pd.read_csv('advertising.csv')
display(tabela)

#Passo 4: Ajuste de Dados (Tratamento/Limpeza)
print(tabela.info())


# #### Análise Exploratória
# - Vamos tentar visualizar como as informações de cada item estão distribuídas
# - Vamos ver a correlação entre cada um dos itens

# In[3]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[6]:


display(tabela.corr()) #observar a linha Vendas.
print('-'*50)

#cria grafico
plt.figure(figsize=(15,5))
sns.heatmap(tabela.corr(), cmap="Wistia", annot=True)
#exibe grafico
plt.show()


# #### Com isso, podemos partir para a preparação dos dados para treinarmos o Modelo de Machine Learning
# 
# - Separando em dados de treino e dados de teste

# In[25]:


from sklearn.model_selection import train_test_split

#y - é quem vc quer prever
y = tabela["Vendas"]

#x = quem eu vou usar para prever y, ou seja, o resto da tabela
x = tabela[["TV", "Radio", "Jornal"]]

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y)


# #### Temos um problema de regressão - Vamos escolher os modelos que vamos usar:
# 
# - Regressão Linear
# - RandomForest (Árvore de Decisão)

# In[26]:


#importa
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

#cria
modelo_regressaolinear = LinearRegression()
modelo_arvoredecisao = RandomForestRegressor()

#treina
modelo_regressaolinear.fit(x_treino, y_treino)
modelo_arvoredecisao.fit(x_treino, y_treino)


# #### Teste da AI e Avaliação do Melhor Modelo
# 
# - Vamos usar o R² -> diz o % que o nosso modelo consegue explicar o que acontece

# In[27]:


from sklearn.metrics import r2_score

#fazer previsoes
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)
previsao_regressaolinear = modelo_regressaolinear.predict(x_teste)

#comparar a previsao com o y_teste
print(r2_score(y_teste, previsao_arvoredecisao))
print(r2_score(y_teste, previsao_regressaolinear))


# #### Visualização Gráfica das Previsões

# In[28]:


tabela_auxiliar = pd.DataFrame()
tabela_auxiliar['y_teste'] = y_teste
tabela_auxiliar['Previsão Regressão Linear'] = previsao_regressaolinear
tabela_auxiliar['Previsão Árvore Decisão'] = previsao_arvoredecisao

plt.figure(figsize=(15,5))
sns.lineplot(data=tabela_auxiliar)
plt.show()


# #### Como fazer uma nova previsão?

# In[31]:


tabela_nova = pd.read_csv("novos.csv")
display(tabela_nova)


# In[32]:


previsao = modelo_arvoredecisao.predict(tabela_nova)
print(previsao)


# ## Qual a importância das variáveis para a Venda?

# In[ ]:


sns.barplot(x=x_treino.columns, y=modelo_arvoredecisao.feature_importances_)
plt.show()

# Caso queira comparar Radio com Jornal
# print(df[["Radio", "Jornal"]].sum())

