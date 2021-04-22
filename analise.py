#!/usr/bin/env python
# coding: utf-8

# In[1]:


###############################################################################################################################
# MLOps -  DevOps + Data Science = MLOps
###############################################################################################################################


# In[10]:


#Intalar pacotes

if False:
    get_ipython().system('pip install pandas')
    get_ipython().system('pip install sklearn')
    get_ipython().system('pip install matplotlib ')


# In[11]:


#Imports 

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, plot_roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[12]:


#Parâmetros

#Semente para números aleatórios
seed = 52


# In[13]:


df_heart = pd.read_csv('SAHeart.csv', index_col=0)
#df_heart.head()


# In[14]:


#df_heart.describe()


# In[15]:


#Dummies
df_heart = pd.get_dummies(df_heart, columns = ['famhist'], drop_first=True)


# In[16]:


#Divisão treino e teste
y = df_heart.pop('chd')
X_train, X_test, y_train, y_test = train_test_split(df_heart, y, test_size=0.25, random_state=seed)


# In[17]:


# Treinar o modelo
model = LogisticRegression(solver='liblinear', random_state=0).fit(X_train, y_train)


# In[18]:


#Avaliação do treino
train_score = model.score(X_train, y_train) * 100

# Avaliação do teste
test_score = model.score(X_test, y_test) * 100


# In[19]:


#Salvar as métrica em um arquivo
with open("metrics.txt", 'w') as outfile:
        outfile.write("variancia_treino: %2.1f%%\n" % train_score)
        outfile.write("variaancia_teste: %2.1f%%\n" % test_score)


# In[23]:


# Confusion Matrix and plot
cm = confusion_matrix(y_test, model.predict(X_test))
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')

plt.tight_layout()
plt.savefig("cm.png",dpi=120) 
plt.close()


# In[21]:


#Imprimir a avaliação do teste
print(classification_report(y_test, model.predict(X_test)))


# In[22]:


#Curva ROC
model_ROC = plot_roc_curve(model, X_test, y_test)
plt.tight_layout()
plt.savefig("roc.png",dpi=120) 
plt.close()


# In[24]:


#Fechar os plots
#plt.close()

