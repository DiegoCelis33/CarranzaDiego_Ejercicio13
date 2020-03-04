#!/usr/bin/env python
# coding: utf-8

# In[17]:


import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import numpy as np
import sklearn.metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import precision_recall_curve
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

# lee los numeros
numeros = skdata.load_digits()

# lee los labels
target = numeros['target']

# lee las imagenes
imagenes = numeros['images']

# cuenta el numero de imagenes total
n_imagenes = len(target)

# para poder correr PCA debemos "aplanar las imagenes"
data = imagenes.reshape((n_imagenes, -1)) # para volver a tener los datos como imagen basta hacer data.reshape((n_imagenes, 8, 8))

# Split en train/test
scaler = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.4)

x_test, x_validation, y_test, y_validation = train_test_split(x_test, y_test, train_size=0.666)

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_validation = scaler.transform(x_validation)





#cov = np.cov(x_train.T)
#valores, vectores = np.linalg.eig(cov)
#valores = np.real(valores)
#vectores = np.real(vectores)
#ii = np.argsort(-valores)
#valores = valores[ii]
#vectores = vectores[:,ii]



n_c = 20
f_1_score = []
numeros = np.arange(0,10)
C_values = np.logspace(0.01,10, n_c)
for i in range(n_c):
#    print(i)
    linear = SVC(C=C_values[i], kernel='linear')
    linear.fit(x_train, y_train)
    y_predict_test = linear.predict(x_test)
    f_1_score.append(f1_score(y_train, y_predict_test, average='macro', labels=numeros))
    
f_1_ = np.array(f_1_score)

maximo_f1 = np.max(f_1_)

print('maximo='+str(np.exp(maximo_f1)))

y_predict_validation = linear.predict(x_validation)

numeros = np.arange(0,10)    
c_matrix = confusion_matrix(y_validation, y_predict_validation, labels = numeros)
plt.figure(figsize=(10,10))
plt.title("C from 0.01 to 10")
plt.imshow(c_matrix)
for i in range(10):
    
    plt.text(i, i, ''+str(c_matrix[i,i]/np.sum(c_matrix[i,:])))
         
plt.savefig("confusion_matrix.png")






# In[26]:


print(x_train.shape)


# In[ ]:




