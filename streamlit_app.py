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
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import joblib

st.title('\U0001F4B5 Предсказание идеального тарифа для клиента')

# Загрузка данных
data_url = "https://raw.githubusercontent.com/your-dataset.csv"
df = pd.read_csv(data_url)

# Вывод исходных данных
with st.expander('Исходные данные'):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("X (Признаки)")
        X_raw = df.drop('ideal_plan', axis=1)
        st.dataframe(X_raw)
    with col2:
        st.subheader("y (Целевая переменная)")
        y_raw = df['ideal_plan']
        st.dataframe(y_raw.to_frame())

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.3, random_state=42)

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
    model_choice = st.selectbox("Выберите модель:", ["Logistic Regression", "Decision Tree", "Random Forest", "CatBoost"])
    
    if model_choice == "Logistic Regression":
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    elif model_choice == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_choice == "Random Forest":
        model = RandomForestClassifier()
    elif model_choice == "CatBoost":
        model = CatBoostClassifier(verbose=0)
    
    model.fit(X_train_scaled, y_train)

# Ввод пользовательских данных
st.title("📞 Предсказание идеального тарифа")

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
    st.subheader("🔮 Идеальный тариф: ")
    st.write(prediction)

# Визуализация важности признаков для CatBoost
if model_choice == "CatBoost":
    importance = model.get_feature_importance()
    fig, ax = plt.subplots()
    ax.barh(X_raw.columns, importance)
    ax.set_xlabel('Важность признаков')
    ax.set_title('Важность признаков для CatBoost')
    st.pyplot(fig)
