# check.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import requests

class FraudDetectionModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
        self.imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.scaler = StandardScaler()
        self.trained = False

    def download_data(self, url, file_path):
        response = requests.get(url)
        with open(file_path, 'wb') as f:
            f.write(response.content)

    def load_data(self, train_path, test_path):
        chunksize = 10 ** 6
        chunks = []
        for chunk in pd.read_csv(train_path, chunksize=chunksize):
            chunks.append(chunk)
        self.df = pd.concat(chunks, axis=0)
        
        chunks_test = []
        for chunk in pd.read_csv(test_path, chunksize=chunksize):
            chunks_test.append(chunk)
        self.df_test = pd.concat(chunks_test, axis=0)
        
        self._preprocess_data()

    def _preprocess_data(self):
        self.df.drop(columns=["Unnamed: 0", "trans_num", "street"], inplace=True)
        data = self.df.head(n=10000)
        df_processed = pd.get_dummies(data=data)
        self.x_train = df_processed.drop(columns='is_fraud', axis=1)
        self.y_train = df_processed['is_fraud']
        
        self.df_test.drop(columns=["Unnamed: 0", "trans_num", "street"], inplace=True)
        data_test = self.df_test.sample(frac=1, random_state=1).reset_index()
        data_test = data_test.head(n=10000)
        df_processed_test = pd.get_dummies(data=data_test)
        self.x_test = df_processed_test.drop(columns='is_fraud', axis=1)
        self.y_test = df_processed_test['is_fraud']
        
        train_cols = self.x_train.columns
        test_cols = self.x_test.columns
        missing_cols = set(train_cols) - set(test_cols)
        for c in missing_cols:
            self.x_test[c] = 0
        self.x_test = self.x_test[train_cols]

        self.x_train_imputed = self.imputer.fit_transform(self.x_train)
        self.x_test_imputed = self.imputer.transform(self.x_test)
        self.x_train_scaled = self.scaler.fit_transform(self.x_train_imputed)
        self.x_test_scaled = self.scaler.transform(self.x_test_imputed)
        
        self.y_train = self.y_train.fillna(self.y_train.mode().iloc[0])
        self.y_test = self.y_test.fillna(self.y_test.mode().iloc[0])

    def train_model(self):
        self.model.fit(self.x_train_scaled, self.y_train)
        self.trained = True

    def predict(self, data):
        if not self.trained:
            raise Exception("Model is not trained yet")
        data_imputed = self.imputer.transform(data)
        data_scaled = self.scaler.transform(data_imputed)
        return self.model.predict(data_scaled)
    
    def evaluate_model(self):
        if not self.trained:
            raise Exception("Model is not trained yet")
        y_pred = self.model.predict(self.x_test_scaled)
        accuracy = accuracy_score(self.y_test, y_pred)
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        class_report = classification_report(self.y_test, y_pred, zero_division=0)
        return accuracy, conf_matrix, class_report
