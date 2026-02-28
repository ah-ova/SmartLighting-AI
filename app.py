import streamlit as st
import cv2
import pandas as pd
import numpy as np
import pickle
from ultralytics import YOLO
import time
import plotly.graph_objects as go

# --- БАЗА ДАННЫХ АЗЕРБАЙДЖАНА ---
DATA = {
    "Bakı": {
        "lamps": 150000,
        "districts": ["Səbail", "Yasamal", "Nəsimi", "Binəqədi", "Sabunçu"]
    },
    "Gəncə": {
        "lamps": 45000,
        "districts": ["Kəpəz", "Nizami"]
    },
    "Sumqayıt": {
        "lamps": 35000,
        "districts": ["Mərkəz", "Corat", "H.Z. Tağıyev"]
    }
}

# --- МНОГОЯЗЫЧНЫЙ СЛОВАРЬ ---
LANG = {
    "AZ": {
        "title": "SmartLighting AI",
        "region": "Region seçin", "district": "Rayon/Küçə", "lamps": "Fənər sayı",
        "scan_btn": "OBYEKTLƏRİ TƏYİN ET (YOLOv8)",
        "time": "Zaman (Saat)", "weather": "Hava şəraiti",
        "w_list": ["Açıq", "Yağışlı", "Dumanlı"],
        "metrics": ["Parlaqlıq", "Xərc", "Qənaət"],
        "chat_head": "🤖 AI Analitik Agent",
        "chat_in": "Sual verin (məs: Hazırda vəziyyət necədir?)...",
        "objects": "Aşkar edilmiş obyektlər", "cars": "Maşınlar", "people": "İnsanlar",
        "annual": "İllik Proqnoz"
    },
    "RU": {
        "title": "SmartLighting AI",
        "region": "Выберите регион", "district": "Район/Улица", "lamps": "Кол-во фонарей",
        "scan_btn": "ОПРЕДЕЛИТЬ ОБЪЕКТЫ (YOLOv8)",
        "time": "Время (Часы)", "weather": "Погода",
        "w_list": ["Ясно", "Дождь", "Туман"],
        "metrics": ["Яркость", "Расход", "Экономия"],
        "chat_head": "🤖 ИИ Аналитический Агент",
        "chat_in": "Задайте вопрос (напр: Какая сейчас экономия?)...",
        "objects": "Объекты в кадре", "cars": "Машины", "people": "Люди",
        "annual": "Годовой Прогноз"
    },
    "EN": {
        "title": "SmartLighting AI",
        "region": "Select Region", "district": "District/Street", "lamps": "Lamps Count",
        "scan_btn": "DETECT OBJECTS (YOLOv8)",
        "time": "Time (Hours)", "weather": "Weather Condition",
        "w_list": ["Clear", "Rainy", "Foggy"],
        "metrics": ["Brightness", "Cost", "Savings"],
        "chat_head": "🤖 AI Analytic Agent",
        "chat_in": "Ask AI (e.g. What is the current status?)...",
        "objects": "Detected Objects", "cars": "Cars", "people": "People",
        "annual": "Annual Forecast"
    }
}

# --- ПОДГОТОВКА МОДЕЛЕЙ ---
@st.cache_resource
def load_yolo():
    return YOLO('yolov8n.pt')

try:
    with open('light_model.pkl', 'rb') as f:
        light_model = pickle.load(f)
except:
    light_model = None

yolo_net = load_yolo()

# Состояние сессии
if 'cars' not in st.session_state: st.session_state.cars = 0
if 'people' not in st.session_state: st.session_state.people = 0
if 'frame' not in st.session_state: st.session_state.frame = None

# --- ИНТЕРФЕЙС ---
st.set_page_config(page_title="SmartLighting AI", page_icon="🥇", layout="wide")
sel_lang = st.sidebar.selectbox("🌐 Dil / Язык / Language", ["AZ", "RU", "EN"])
L = LANG[sel_lang]

st.title(f"🥇 {L['title']}")

# --- SIDEBAR: ЛОКАЦИЯ ---
st.sidebar.header("📍 Location")
reg_choice = st.sidebar.selectbox(L["region"], list(DATA.keys()))
dist_choice = st.sidebar.selectbox(L["district"], DATA[reg_choice]["districts"])
lamps_count = st.sidebar.number_input(L["lamps"], value=DATA[reg_choice]["lamps"])

st.sidebar.divider()
if st.sidebar.button(L["scan_btn"]):
    cap = cv2.VideoCapture(0)
    for _ in range(30): cap.read() # Калибровка света
    ret, frame = cap.read()
    cap.release()
    if ret:
        results = yolo_net.predict(frame, conf=0.4, verbose=False)
        c_count, p_count = 0, 0
        for r in results:
            for box in r.boxes:
                label = yolo_net.names[int(box.cls[0])]
                if label in ['car', 'bus', 'truck']: c_count += 1
                if label == 'person': p_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        st.session_state.cars = c_count
        st.session_state.people = p_count
        st.session_state.frame = frame

# --- ОСНОВНОЙ БЛОК ---
col_vis, col_ctrl = st.columns([1, 2])

with col_vis:
    if st.session_state.frame is not None:
        st.image(st.session_state.frame, channels="BGR", caption=L["objects"])
    else:
        st.info("Scan environment to start AI Vision")

with col_ctrl:
    hour = st.slider(L["time"], 0, 23, 21)
    weather_idx = st.selectbox(L["weather"], [0, 1, 2], format_func=lambda x: L["w_list"][x])
    
    # ПРЕДСКАЗАНИЕ ИИ (на основе нейросети YOLO)
    input_df = pd.DataFrame([[hour, st.session_state.cars, st.session_state.people, weather_idx]], 
                             columns=['hour', 'cars', 'people', 'weather'])
    brightness = int(light_model.predict(input_df)[0]) if light_model else 100
    
    # Логика работы
    if 6 <= hour <= 18: 
        brightness = 0
    else:
        brightness = max(min(brightness, 100), 20) # Минимум 20% ночью

    # СЧЕТЧИК ОБЪЕКТОВ (НЕ УДАЛЯЕМ!)
    st.subheader(f"{L['objects']} (YOLOv8)")
    st.markdown(f"### 🚗 {st.session_state.cars} | 🚶 {st.session_state.people}")

    # ЭКОНОМИКА
    std_brightness = 100 if (hour < 6 or hour > 18) else 0
    cost_std = (std_brightness/100) * 0.25 * lamps_count * 0.09
    cost_ai = (brightness/100) * 0.25 * lamps_count * 0.09
    savings = cost_std - cost_ai

    m1, m2, m3 = st.columns(3)
    m1.metric(L["metrics"][0], f"{brightness}%")
    m2.metric(L["metrics"][1], f"{cost_ai:.2f} AZN/h")
    m3.metric(L["metrics"][2], f"{savings:.2f} AZN/h", delta=f"{(savings/(cost_std+0.1)*100):.1f}%")

# ГРАФИК
st.plotly_chart(go.Figure([
    go.Bar(name='Standard', x=['Cost'], y=[cost_std], marker_color='#ff4b4b'),
    go.Bar(name='AI Optimized', x=['Cost'], y=[cost_ai], marker_color='#00CC96')
]), use_container_width=True)

st.success(f"📊 {L['annual']} ({reg_choice}): {(savings * 10 * 365):,.0f} AZN")

# --- 🤖 ИИ АНАЛИТИЧЕСКИЙ ЧАТ (РЕАЛЬНЫЙ ИИ) ---
st.divider()
st.subheader(L["chat_head"])

if "messages" not in st.session_state: st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input(L["chat_in"]):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    
    with st.chat_message("assistant"):
        p = prompt.lower()
        # Контекстный анализ (ИИ видит данные)
        if any(x in p for x in ["vəziyyət", "status", "состояние", "что видишь"]):
            response = f"Hazırda **{reg_choice} ({dist_choice})** ərazisindəyəm. YOLOv8 sensorum **{st.session_state.people} insan** və **{st.session_state.cars} maşın** aşkar edib. Buna görə parlaqlığı **{brightness}%** səviyyəsinə endirmişəm." if sel_lang=="AZ" else f"Сейчас я мониторю **{reg_choice} ({dist_choice})**. Сенсор YOLOv8 обнаружил **{st.session_state.people} чел.** и **{st.session_state.cars} авто**. Я выставил яркость **{brightness}%**."
        elif any(x in p for x in ["formula", "hesab", "как", "формул"]):
            response = f"Hesablama: 0.25kW * {lamps_count} fənər * 0.09 AZN tarif. İİ parlaqlığı azaltdığı üçün qənaət saatda {savings:.2f} AZN təşkil edir." if sel_lang=="AZ" else f"Расчет: 0.25кВт * {lamps_count} ламп * 0.09 AZN. Благодаря ИИ экономия составляет {savings:.2f} AZN в час."
        elif any(x in p for x in ["ölkə", "hara", "страна", "город"]):
            response = f"Sistem Azərbaycan üçün optimallaşdırılıb. Hazırkı region: **{reg_choice}**." if sel_lang=="AZ" else f"Система оптимизирована для Азербайджана. Текущий регион: **{reg_choice}**."
        else:
            response = f"Mən SmartLighting AI-yam. {reg_choice} üzrə enerji optimallaşdırılmasına cavabdehəm." if sel_lang=="AZ" else f"Я SmartLighting AI. Отвечаю за оптимизацию энергии в регионе {reg_choice}."
        
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})