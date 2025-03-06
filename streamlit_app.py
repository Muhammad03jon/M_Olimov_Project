import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import joblib

st.title('📞 Предсказание идеального тарифа для клиента')

# Загрузка данных
data_url = "https://raw.githubusercontent.com/Muhammad03jon/M_Olimov_Project/refs/heads/master/data.csv"
df = pd.read_csv(data_url)

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

# Проверка на пропуски в данных
if df.isnull().any().any():
    st.warning("В данных присутствуют пропущенные значения. Выполняется их заполнение.")
    # Заполнение пропусков: числовые признаки - средним значением, категориальные - наиболее частым значением
    df = df.fillna(df.mean(numeric_only=True))
    df = df.apply(lambda x: x.fillna(x.mode()[0]) if x.dtype == 'object' else x, axis=0)

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

# Выбор модели и гиперпараметров
with st.sidebar:
    st.header("Выбор модели")
    model_choice = st.selectbox("Выберите модель:", ["Logistic Regression", "Decision Tree", "Random Forest"])
    
    if model_choice == "Logistic Regression":
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    elif model_choice == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_choice == "Random Forest":
        model = RandomForestClassifier()
    
    model.fit(X_train_scaled, y_train)

# Ввод пользовательских данных
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
OnlineBackup = st.sidebar.radio("Резервное копирование?", ["Yes", "No", "No internet service"])
DeviceProtection = st.sidebar.radio("Защита устройств?", ["Yes", "No", "No internet service"])
TechSupport = st.sidebar.radio("Техподдержка?", ["Yes", "No", "No internet service"])
StreamingTV = st.sidebar.radio("Стриминг ТВ?", ["Yes", "No", "No internet service"])
StreamingMovies = st.sidebar.radio("Стриминг фильмов?", ["Yes", "No", "No internet service"])

# Контракт
Contract = st.sidebar.selectbox("Тип контракта:", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.sidebar.radio("Безбумажный биллинг?", ["Yes", "No"])

# Способ оплаты
PaymentMethod = st.sidebar.selectbox("Способ оплаты:", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

# Вывод введенных данных
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
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
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
    
    prediction = model.predict(input_df)[0]
    predicted_class = class_mapping[prediction]  # Получаем строковое значение из числа

    st.subheader("🔮 Идеальный тариф: ")
    st.write(predicted_class)

    # Если модель поддерживает вероятности для каждого класса
    if model_choice in ["Logistic Regression", "Random Forest", "Decision Tree"]:
        prediction_prob = model.predict_proba(input_df)[0]
        st.write("Предсказанные вероятности для каждого тарифа:")
        for i, prob in enumerate(prediction_prob):
            st.write(f"Тариф {class_mapping[i]}: {prob:.2f}")

# Визуализация важности признаков для дерева решений или случайного леса
if model_choice in ["Decision Tree", "Random Forest"]:
    importance = model.feature_importances_
    fig, ax = plt.subplots()
    ax.barh(X_raw.columns, importance)
    ax.set_xlabel('Важность признаков')
    ax.set_title(f'Важность признаков для {model_choice}')
    st.pyplot(fig)

# Отображение метрик
if st.button("Метрики"):
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    st.subheader("📊 Метрики модели")

    # Вывод метрик
    st.write("Точность (Accuracy):", model.score(X_test_scaled, y_test))
    st.write("Прецизионность (Precision):", classification_report(y_test, y_pred, target_names=list(class_mapping.values())))
    st.write("Полнота (Recall):", classification_report(y_test, y_pred, target_names=list(class_mapping.values())))
    st.write("F1-Score:", classification_report(y_test, y_pred, target_names=list(class_mapping.values())))

    # ROC-AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    st.write(f"ROC AUC: {roc_auc:.2f}")

    # Матрица ошибок
    st.subheader("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=list(class_mapping.values()), yticklabels=list(class_mapping.values()))
    st.pyplot()
