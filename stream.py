import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Titanic veri setini yükleme
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Veri ön işleme
# Null değerleri doldurma ve gereksiz sütunları kaldırma
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Kategorik değişkenleri kodlama
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Normalizasyon
scaler = StandardScaler()
X = scaler.fit_transform(df.drop('Survived', axis=1))
y = df['Survived']

# Özellik seçimi
anova_selector = SelectKBest(score_func=f_classif, k=4)
X_selected = anova_selector.fit_transform(X, y)

# Eğitim ve test verisi olarak bölümleme
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Modelleme
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Tahmin
y_pred = model.predict(X_test)

# Model değerlendirme
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("Confusion Matrix:\n", conf_matrix)
