"""retrain_light.py
Rápido retrain usando RandomForest no CSV aumentado e salva os artefactos em models/.
"""
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

INPUT_CSV = 'hand_landmarks_augmented.csv'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

print('Carregando dataset:', INPUT_CSV)
df = pd.read_csv(INPUT_CSV)
cols = [c for c in df.columns if c != 'label']
X = df[cols].values
y = df['label'].values

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print('Treinando RandomForest (light)...')
model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
model.fit(X_train_s, y_train)

# Avaliação
y_pred = model.predict(X_test_s)
acc = accuracy_score(y_test, y_pred)
print('Accuracy test:', acc)
print(classification_report(y_test, y_pred, zero_division=0))

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=False, fmt='d')
plt.title('Confusion matrix - retrain_light')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'confusion_retrain_light.png'))
plt.close()

# Save artifacts
joblib.dump(model, os.path.join(MODEL_DIR, 'best_model.pkl'))
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler_hand_sign.pkl'))
joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.pkl'))
print('Artefactos guardados em', MODEL_DIR)
