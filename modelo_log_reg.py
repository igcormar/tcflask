import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
import pickle
import os



data = pd.read_csv("./data/online_gaming_behavior_dataset.csv")



#instanciamos lista de columnas con las que entrenamos
num_col = ['SessionsPerWeek', 'AvgSessionDurationMinutes', 'AchievementsUnlocked']

#instanciamos el target
target = 'EngagementLevel'


#dividimos en train y test
train_set, test_set = train_test_split(data, test_size=0.2, stratify=data[target], random_state=42)

X_train = train_set[num_col]
X_test = test_set[num_col]

y_train = train_set[target]
y_test = test_set[target]


#mapearemos 'y' para entrenar
labels = {'Low':0,
          'Medium': 1,
          'High': 2}
y_train_encoded = y_train.map(labels)


#Preprocessing
num_pipe = Pipeline([
    ('mmscaler', MinMaxScaler())
])

preprocessing = ColumnTransformer([
    ('normalize', num_pipe, num_col)],
    remainder='passthrough').set_output(transform= 'pandas')


#Entrenamiento
logreg_pipe = Pipeline([
    ('preprocessing', preprocessing),
    ('modelo', LogisticRegression(class_weight='balanced', C=1.0, max_iter=50, 
                                  solver='saga', random_state=42))])

logreg_pipe.fit(X_train, y_train_encoded)

with open('./model/Modelo_Logistic_regression', 'wb') as file:
    pickle.dump(logreg_pipe, file)






