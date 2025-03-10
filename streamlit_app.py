import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedKFold, cross_val_score

st.title('📞 Предсказание идеального тарифа для клиента')

# Загрузка данных
data_url = "https://raw.githubusercontent.com/Muhammad03jon/M_Olimov_Project/refs/heads/master/data.csv"
df = pd.read_csv(data_url)
df = df.drop(columns=['OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'])

# Словарь для преобразования числовых значений в соответствующие классы
class_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}

# Вывод исходных данных
with st.expander('Исходные данные'):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("X (Признаки)")
        X_raw = df.drop('ideal_plan', axis=1)
        st.dataframe(X_raw)
    with col2:
        st.subheader("y (Целевая переменная)")
        y_raw = df['ideal_plan']  # Переход от строковых значений к числам
        st.dataframe(y_raw.to_frame())

# Преобразуем целевую переменную для обучения
df['ideal_plan'] = df['ideal_plan'].map({'Low': 0, 'Medium': 1, 'High': 2})

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X_raw, df['ideal_plan'], test_size=0.2, random_state=42)

# Кодирование категориальных признаков
categorical_features = X_raw.select_dtypes(include=['object']).columns
encoder = TargetEncoder(cols=categorical_features)
X_train = encoder.fit_transform(X_train, y_train)
X_test = encoder.transform(X_test)

# Стандартизация данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

oversampler = RandomOverSampler(random_state=42)
X_train_scaled, y_train = oversampler.fit_resample(X_train_scaled, y_train)

best_model = RandomForestClassifier(
    class_weight='balanced',
    max_depth=5, 
    min_samples_leaf=3, 
    min_samples_split=10, 
    n_estimators=50,
    random_state=42
)

# Обучаем модель
best_model.fit(X_train_scaled, y_train)

# Ввод данных
st.sidebar.header("Введите данные клиента")

# Пол пользователя
gender = st.sidebar.selectbox("Пол:", ["Male", "Female"])

# Возраст и статус пенсионера
age = st.sidebar.slider("Возраст:", 18, 100, 30)
SeniorCitizen = 1 if age >= 65 else 0

# Семейное положение
Partner = st.sidebar.radio("Есть партнер?", ["Yes", "No"])
Dependents = st.sidebar.radio("Есть иждивенцы?", ["Yes", "No"])

# Стаж использования услуг
tenure = st.sidebar.slider("Стаж (месяцы):", 0, 72, 12)

# Услуги связи
PhoneService = st.sidebar.radio("Подключена телефонная связь?", ["Yes", "No"])
MultipleLines = st.sidebar.radio("Несколько линий?", ["Yes", "No phone service", "No"])

# Интернет-услуги
OnlineSecurity = st.sidebar.radio("Защита в интернете?", ["Yes", "No", "No internet service"])

# Контракт
Contract = st.sidebar.selectbox("Тип контракта:", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.sidebar.radio("Безбумажный биллинг?", ["Yes", "No"])

# Способ оплаты
PaymentMethod = st.sidebar.selectbox("Способ оплаты:", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

# Вывод данных
st.subheader("📝 Введенные данные клиента")
input_data = {
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "OnlineSecurity": OnlineSecurity,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod
}
input_df = pd.DataFrame([input_data])
st.dataframe(input_df)

# Предсказание тарифа
if st.button("Предсказать"):
    input_df = pd.DataFrame([input_data])
    input_df = encoder.transform(input_df)
    input_df = scaler.transform(input_df)
    
    prediction = best_model.predict(input_df)[0]
    predicted_class = class_mapping[prediction]  # Получаем строковое значение из числа

    st.subheader("🔮 Идеальный тариф: ")
    st.write(predicted_class)

    # Если модель поддерживает вероятности для каждого класса
    prediction_prob = best_model.predict_proba(input_df)[0]
    st.write("Предсказанные вероятности для каждого тарифа:")
    for i, prob in enumerate(prediction_prob):
        st.write(f"Тариф {class_mapping[i]}: {prob:.2f}")

# Отображение метрик
if st.button("Метрики"):
    y_pred_train = best_model.predict(X_train_scaled)
    y_pred_test = best_model.predict(X_test_scaled)

    y_pred_proba_train = best_model.predict_proba(X_train_scaled)
    y_pred_proba_test = best_model.predict_proba(X_test_scaled)

    st.subheader("📊 Метрики модели")

    # Accuracy для обучения и теста
    accuracy_train = best_model.score(X_train_scaled, y_train)
    accuracy_test = best_model.score(X_test_scaled, y_test)
    st.write(f"**Точность (Accuracy) на обучающем наборе:** {accuracy_train:.2f}")
    st.write(f"**Точность (Accuracy) на тестовом наборе:** {accuracy_test:.2f}")

    # ROC-AUC
    roc_auc_train = roc_auc_score(y_train, y_pred_proba_train, multi_class="ovr")
    roc_auc_test = roc_auc_score(y_test, y_pred_proba_test, multi_class="ovr")

    st.write(f"**ROC AUC на обучающем наборе:** {roc_auc_train:.2f}")
    st.write(f"**ROC AUC на тестовом наборе:** {roc_auc_test:.2f}")

    # Метрики Precision, Recall, F1-Score
    precision = precision_score(y_test, y_pred_test, average='macro')
    recall = recall_score(y_test, y_pred_test, average='macro')
    f1 = f1_score(y_test, y_pred_test, average='macro')

    metrics_df = pd.DataFrame({
        'Прецизионность (Precision)': precision,
        'Полнота (Recall)': recall,
        'F1-Оценка': f1
    }, index=['Low', 'Medium', 'High'])

    st.write("Метрики для каждого класса:")
    st.dataframe(metrics_df)

    # Матрица ошибок
    st.subheader("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_test)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
    ax.set_title('Матрица ошибок')
    st.pyplot(fig)
    # Создаём объект StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Используем кросс-валидацию
    cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=skf, scoring='accuracy')
    
    # Выводим результаты
    st.subheader("📊 Результаты кросс-валидации")
    st.write(f"Средняя точность (Accuracy): {cv_scores.mean():.4f}")
    st.write(f"Стандартное отклонение: {cv_scores.std():.4f}")
    st.write("Результаты по фолдам:", cv_scores)
