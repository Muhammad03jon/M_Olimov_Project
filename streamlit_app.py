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

# Заголовок приложения
st.title('📞 Предсказание идеального тарифа для клиента')

# Описание приложения
st.markdown("""
    Это приложение использует модель машинного обучения для предсказания, какой тарифный план лучше всего подходит для клиента телекоммуникационной компании.
    Используемые данные позволяют предсказать тарифный план на основе различных признаков клиента.
""")

# Загрузка данных
data_url = "https://raw.githubusercontent.com/Muhammad03jon/M_Olimov_Project/refs/heads/master/data.csv"
df = pd.read_csv(data_url)

# Преобразуем целевую переменную для обучения
df['ideal_plan'] = df['ideal_plan'].map({'Low': 0, 'Medium': 1, 'High': 2})

# Разделение данных на признаки и целевую переменную
X_raw = df.drop('ideal_plan', axis=1)
y_raw = df['ideal_plan']

# Кодирование категориальных признаков
categorical_features = X_raw.select_dtypes(include=['object']).columns
encoder = TargetEncoder(cols=categorical_features)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

# Применяем кодирование
X_train = encoder.fit_transform(X_train, y_train)
X_test = encoder.transform(X_test)

# Удаляем сильно коррелирующие признаки, оставляем только TechSupport
drop_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'StreamingTV', 'StreamingMovies']
X_train = X_train.drop(columns=drop_cols)
X_test = X_test.drop(columns=drop_cols)

# Стандартизация данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Балансировка классов
oversampler = RandomOverSampler(random_state=42)
X_train_scaled, y_train = oversampler.fit_resample(X_train_scaled, y_train)

# Обучение модели
best_model = RandomForestClassifier(
    class_weight='balanced',
    max_depth=6, 
    min_samples_leaf=3, 
    min_samples_split=10, 
    n_estimators=50,
    random_state=42
)
best_model.fit(X_train_scaled, y_train)

# Вкладка для вывода метрик модели
st.sidebar.subheader("📊 Метрики модели")

# Вычисление метрик
accuracy_train = best_model.score(X_train_scaled, y_train)
accuracy_test = best_model.score(X_test_scaled, y_test)
y_pred_proba_train = best_model.predict_proba(X_train_scaled)
y_pred_proba_test = best_model.predict_proba(X_test_scaled)
roc_auc_train = roc_auc_score(y_train, y_pred_proba_train, multi_class="ovr")
roc_auc_test = roc_auc_score(y_test, y_pred_proba_test, multi_class="ovr")
y_pred_test = best_model.predict(X_test_scaled)
precision = precision_score(y_test, y_pred_test, average='macro')
recall = recall_score(y_test, y_pred_test, average='macro')
f1 = f1_score(y_test, y_pred_test, average='macro')

# Вывод метрик в сайдбаре
st.sidebar.write(f"**Точность (Accuracy) на обучающем наборе:** {accuracy_train:.2f}")
st.sidebar.write(f"**Точность (Accuracy) на тестовом наборе:** {accuracy_test:.2f}")
st.sidebar.write(f"**ROC AUC на обучающем наборе:** {roc_auc_train:.2f}")
st.sidebar.write(f"**ROC AUC на тестовом наборе:** {roc_auc_test:.2f}")
st.sidebar.write(f"**Прецизионность (Precision):** {precision:.2f}")
st.sidebar.write(f"**Полнота (Recall):** {recall:.2f}")
st.sidebar.write(f"**F1-Оценка:** {f1:.2f}")

# Матрица ошибок
st.subheader("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_test)
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
ax.set_title('Матрица ошибок')
st.pyplot(fig)

# ROC-кривые
st.subheader("ROC-кривые для каждого класса")
fig, ax = plt.subplots(figsize=(7, 5))
for i in range(3):
    fpr, tpr, _ = roc_curve(y_test == i, y_pred_proba_test[:, i])
    ax.plot(fpr, tpr, label=f"Class {class_mapping[i]}")
ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC-кривые")
ax.legend()
st.pyplot(fig)

# Корреляционная матрица
st.subheader("🔗 Корреляция признаков (после стандартизации)")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(pd.DataFrame(X_train_scaled).corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Ввод новых данных
st.subheader("🔧 Ввод новых данных для предсказания")

# Панель ввода данных
new_data = {}

# Пример полей ввода для новых данных
columns = X_raw.columns
for col in columns:
    if X_raw[col].dtype == 'object':
        new_data[col] = st.selectbox(col, options=df[col].unique())
    else:
        new_data[col] = st.number_input(col, value=0.0)

# Преобразуем данные в DataFrame
new_data_df = pd.DataFrame([new_data])

# Кодируем и стандартизируем новые данные
new_data_encoded = encoder.transform(new_data_df)
new_data_scaled = scaler.transform(new_data_encoded)

# Предсказание
if st.button("Предсказать идеальный тариф"):
    prediction = best_model.predict(new_data_scaled)
    st.write(f"Предсказанный идеальный тариф: {class_mapping[prediction[0]]}")
