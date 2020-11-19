# -*- coding: utf-8 -*-
"""

Sript para competencia en Hackathon 2020

Andrés Rogers, andres.rogers@gmail.com

Attribute Description:
    
Invoice No - Invoice ID, encoded as Label
StockCode - Unique code per stock, encoded as Label
Description - The Description, encoded as Label
Quantity - Quantity purchased
InvoiceDate - Date of purchase
UnitPrice - The target value, price of every product
CustomerID - Unique Identifier for every Customer
Country - Country of sales, encoded as Label

"""

# Modelo de Regresión Lineal Múltiple

#%%

# Importar las librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc

# Importar el data set
dataset = pd.read_csv('Train.csv')

"""
# Convertir columna de fechas en valores numéricos
import datetime as dt
dataset['InvoiceDate'] = dataset['InvoiceDate'].apply( lambda x: dt.datetime.strptime(x,'%Y-%m-%d %H:%M:%S') )
dataset['InvoiceDate'] = dataset['InvoiceDate'].apply( lambda x: x.timestamp() )

# Check for missing values
dataset.isnull().sum() # No tiene
"""
"""
# Check for outliers
sns.distplot(dataset['InvoiceNo'])
sns.distplot(dataset['StockCode'])
sns.distplot(dataset['Description'])
sns.distplot(dataset['Quantity']) # Remover Outliers
sns.distplot(dataset['InvoiceDate'])
sns.distplot(dataset['CustomerID'])
sns.distplot(dataset['Country']) 

q_low = dataset['Quantity'].quantile(0.003) # -3 Standard Deviations
q_high = dataset['Quantity'].quantile(0.997) # +3 Standard Deviations
dataset = dataset[ dataset['Quantity'] > q_low ] # Remove lower outliers
dataset = dataset[ dataset['Quantity'] < q_high ] # Remove upper outliers
"""
"""
# Check for multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
variables = dataset[['InvoiceNo','StockCode','Description','Quantity','InvoiceDate','CustomerID','Country']]
vif = pd.DataFrame()
df = add_constant(variables)
vif['VIF'] = [ variance_inflation_factor(df.values, i) for i in range(df.shape[1]) ]
vif['features'] = df.columns
vif # InvoiceNo and InvoiceDate both have Strong Multicollinearity (Values >5 or >10) => Remove InvoiceNo
"""

# Take a smaller part of the Dataset (too big)
#dataset = dataset.head(100000)

# Separar dataset entre variables dependientes (X) y dependiente (y)
X = dataset.iloc[:, [1] ].values # Variables independientes, Remover InvoiceDate
y = dataset.iloc[:, 5].values # Variable dependiente a predecir


del dataset
del df
del variables
del vif
del q_high
del q_low
gc.collect()

#%%

top_sc = dataset.StockCode.value_counts().sort_values(ascending = False).head(500)



#%%
"""
df = pd.get_dummies(dataset, columns=['StockCode','Description','Country'])
X = df
"""
# Label and OneHotEncode Categorical Values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer

#X = X.astype('int') # Chosen variables are all int

labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = make_column_transformer((OneHotEncoder(), [0]), remainder = "passthrough")
X = onehotencoder.fit_transform(X).toarray()

X = X.astype('int')

# Evitar problema variables Dummy
X = X[:,1:]

del labelencoder_X
del onehotencoder
gc.collect()


# Dividir el dataset en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

del X
del y
gc.collect()


#%%
"""
# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

del sc_X
gc.collect()
"""
#%%

# Ajustar el modelo de Regresión lineal múltiple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)
y_pred = regression.predict(X_test) # Predicción de los resultados en el conjunto de testing

gc.collect()

#%%

# Calculate the root mean square error (RMSE) for training data

rmse = np.sqrt(np.square(np.subtract(y_test,y_pred)).mean())
print(rmse)


#%%

# Importar el data set de Testing
final_test = pd.read_csv('Test.csv')
X = final_test.iloc[:, [1] ].values
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = make_column_transformer((OneHotEncoder(), [0]), remainder = "passthrough")
X = onehotencoder.fit_transform(X).toarray()

y_final = regression.predict(X) # Predicción de los resultados en el conjunto de testing

y_final = pd.DataFrame(y_final)
y_final.columns = ['UnitPrice']
y_final.to_csv('enviar.csv',index=False)

del final_test                             
del labelencoder_X                        
del onehotencoder
gc.collect()

#%%

# Construir el modelo óptimo de RLM utilizando la Eliminación hacia atrás
import statsmodels.api as sm
X = np.append(arr = np.ones((282835,1)).astype(int), values = X, axis = 1)
SL = 0.05

#X_opt = X[:, [0,1,2]]
X_opt = X
regression_OLS = sm.OLS(endog = y, exog = X_opt.tolist()).fit()
regression_OLS.summary()





















