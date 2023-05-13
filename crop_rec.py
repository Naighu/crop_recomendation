import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report



df = pd.read_csv('Crop_recommendation.csv')
y = df['label']
X = df.drop(['label'], axis=1)
min_max = MinMaxScaler()
X = min_max.fit_transform(X)

#Neural network
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(4,), activation='tanh', solver='adam', learning_rate='adaptive')
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
mlp.fit(X_train, y_train)
print("MLP trainig completed")

#SVC
from sklearn.svm import SVC

svc = SVC(probability=True)
svc.fit(X_train, y_train)
print("SVC trainig completed")

#RandomForest
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
print("Random forest trainig completed")

#Naiv Bayes
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print("Naive bayes trainig completed")

#XGBOOST
# from xgboost import XGBClassifier
# xgb = XGBClassifier()
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# y_xgb = le.fit_transform(y_train)
# xgb.fit(X_train, y_xgb)
# print("XGboost trainig completed")


def predict(model,input):
    p = model.predict_proba(input)
    # print(p)
    predicted = list(zip(p[0], model.classes_))
    predicted.sort(reverse=True)
    return predicted

def predict_xg(input):
    xgb_preds = xgb.predict(X_test)
    xgb_classes = le.classes_[xgb_preds]
    return xgb_classes

input = np.random.random((1,7))

# results = predict(mlp,input)
# print(f"MLP prdiction {results[:5]}")

# results = predict(svc,input)
# print(f"SVC prdiction {results[:5]}")

# results = predict(rfc,input)
# print(f"RFC prdiction {results[:5]}")

# results = predict(gnb,input)
# print(f"GNB prdiction {results[:5]}")

# results = predict_xg(input)
# print(f"XGBOOST prdiction {results[:5]}")
