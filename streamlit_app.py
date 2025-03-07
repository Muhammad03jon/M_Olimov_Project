import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve
from imblearn.over_sampling import RandomOverSampler

st.title('📞 Предсказание идеального тарифа для клиента')

# Загрузка данных
data_url = "https://raw.githubusercontent.com/Muhammad03jon/M_Olimov_Project/refs/heads/master/data.csv"
df = pd.read_csv(data_url)

# Словарь для преобразования числовых значений в соответствующие классы
class_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}

# Преобразуем целевую переменную для обучения
df['ideal_plan'] = df['ideal_plan'].map({'Low': 0, 'Medium': 1, 'High': 2})

# Разделение данных
X_raw = df.drop('ideal_plan', axis=1)
y_raw = df['ideal_plan']

# Кодирование категориальных признаков
categorical_features = X_raw.select_dtypes(include=['object']).columns
encoder = TargetEncoder(cols=categorical_features)
X_raw = encoder.fit_transform(X_raw, y_raw)

# Вывод корреляции всех признаков
st.subheader("🔗 Корреляция признаков")
correlation = X_raw.corr()
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Удаляем сильно коррелирующие признаки (к примеру, 'OnlineSecurity', 'OnlineBackup', и т.д.)
drop_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'StreamingTV', 'StreamingMovies']
X_raw = X_raw.drop(columns=drop_cols)

# Выводим обновленную корреляцию после удаления колонок
st.subheader("🔗 Обновленная корреляция признаков")
correlation_updated = X_raw.corr()
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(correlation_updated, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

# Стандартизация данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обработка дисбаланса классов
oversampler = RandomOverSampler(random_state=42)
X_train_scaled, y_train = oversampler.fit_resample(X_train_scaled, y_train)

# Обучаем модель RandomForest
best_model = RandomForestClassifier(
    class_weight='balanced',
    max_depth=6, 
    min_samples_leaf=3, 
    min_samples_split=10, 
    n_estimators=50,
    random_state=42
)

# Обучаем модель
best_model.fit(X_train_scaled, y_train)

# Вывод метрик
st.subheader("📊 Метрики модели")

# Accuracy
accuracy_train = best_model.score(X_train_scaled, y_train)
accuracy_test = best_model.score(X_test_scaled, y_test)
st.write(f"**Точность (Accuracy) на обучающем наборе:** {accuracy_train:.2f}")
st.write(f"**Точность (Accuracy) на тестовом наборе:** {accuracy_test:.2f}")

# ROC-AUC
y_pred_proba_train = best_model.predict_proba(X_train_scaled)
y_pred_proba_test = best_model.predict_proba(X_test_scaled)
roc_auc_train = roc_auc_score(y_train, y_pred_proba_train, multi_class="ovr")
roc_auc_test = roc_auc_score(y_test, y_pred_proba_test, multi_class="ovr")
st.write(f"**ROC AUC на обучающем наборе:** {roc_auc_train:.2f}")
st.write(f"**ROC AUC на тестовом наборе:** {roc_auc_test:.2f}")

# Precision, Recall, F1-Score
y_pred_test = best_model.predict(X_test_scaled)
precision = precision_score(y_test, y_pred_test, average='macro')
recall = recall_score(y_test, y_pred_test, average='macro')
f1 = f1_score(y_test, y_pred_test, average='macro')

metrics_df = pd.DataFrame({
    'Прецизионность (Precision)': [precision],
    'Полнота (Recall)': [recall],
    'F1-Оценка': [f1]
})
st.write("Метрики модели:")
st.dataframe(metrics_df)

# Матрица ошибок
st.subheader("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_test)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
ax.set_title('Матрица ошибок')
st.pyplot(fig)

# ROC-кривые
st.subheader("ROC-кривые для каждого класса")
fig, ax = plt.subplots()
for i in range(3):
    fpr, tpr, _ = roc_curve(y_test == i, y_pred_proba_test[:, i])
    ax.plot(fpr, tpr, label=f"{class_mapping[i]}")
ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC-кривые")
ax.legend()
st.pyplot(fig)
