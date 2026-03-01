import streamlit as st
import cv2
import pandas as pd
import numpy as np
import pickle
from ultralytics import YOLO
import time
import plotly.graph_objects as go
import google.generativeai as genai # Подключаем настоящий ИИ

# --- ИНИЦИАЛИЗАЦИЯ НЕЙРОСЕТИ GEMINI ---
try:
    # Ключ берется из файла .streamlit/secrets.toml
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    has_real_ai = True
except Exception:
    has_real_ai = False

# --- БАЗА ДАННЫХ РЕГИОНОВ АЗЕРБАЙДЖАНА ---
DATA = {
    "Bakı": {"lamps": 150000, "districts": ["Səbail", "Yasamal", "Nəsimi", "Binəqədi", "Sabunçu"]},
    "Gəncə": {"lamps": 45000, "districts": ["Kəpəz", "Nizami"]},
    "Sumqayıt": {"lamps": 35000, "districts": ["Mərkəz", "Corat", "H.Z. Tağıyev"]}
}

# --- МНОГОЯЗЫЧНЫЙ СЛОВАРЬ ---
LANG = {
    "AZ": {
        "title": "SmartLighting AI: Neural Edition",
        "region": "Region seçin", "district": "Rayon/Küçə", "lamps": "Fənər sayı",
        "scan_btn": "OBYEKTLƏRİ TƏYİN ET (YOLOv8)",
        "time": "Zaman (Saat)", "weather": "Hava şəraiti",
        "w_list": ["Açıq", "Yağışlı", "Dumanlı"],
        "metrics": ["Parlaqlıq", "Xərc", "Qənaət"],
        "chat_head": "🤖 Real AI Analitik (Gemini 1.5)",
        "chat_in": "İİ-yə sual verin...",
        "objects": "Aşkar edilmiş obyektlər"
    },
    "RU": {
        "title": "SmartLighting AI: Neural Edition",
        "region": "Выберите регион", "district": "Район/Улица", "lamps": "Кол-во фонарей",
        "scan_btn": "ОПРЕДЕЛИТЬ ОБЪЕКТЫ (YOLOv8)",
        "time": "Время (Часы)", "weather": "Погода",
        "w_list": ["Ясно", "Дождь", "Туман"],
        "metrics": ["Яркость", "Расход", "Экономия"],
        "chat_head": "🤖 Настоящий ИИ Аналитик (Gemini 1.5)",
        "chat_in": "Задайте вопрос ИИ...",
        "objects": "Объекты в кадре"
    },
    "EN": {
        "title": "SmartLighting AI: Neural Edition",
        "region": "Select Region", "district": "District/Street", "lamps": "Lamps Count",
        "scan_btn": "DETECT OBJECTS (YOLOv8)",
        "time": "Time (Hours)", "weather": "Weather Condition",
        "w_list": ["Clear", "Rainy", "Foggy"],
        "metrics": ["Brightness", "Cost", "Savings"],
        "chat_head": "🤖 Real AI Analytic Agent (Gemini 1.5)",
        "chat_in": "Ask AI anything...",
        "objects": "Detected Objects"
    }
}

# --- ЗАГРУЗКА МОДЕЛЕЙ ---
@st.cache_resource
def load_yolo():
    return YOLO('yolov8n.pt')

try:
    with open('light_model.pkl', 'rb') as f:
        light_model = pickle.load(f)
except:
    light_model = None

yolo_net = load_yolo()

# Состояние сессии (Память сайта)
if 'cars' not in st.session_state: st.session_state.cars = 0
if 'people' not in st.session_state: st.session_state.people = 0
if 'frame' not in st.session_state: st.session_state.frame = None
if "messages" not in st.session_state: st.session_state.messages = []

# --- ИНТЕРФЕЙС ---
st.set_page_config(page_title="SmartLighting AI", page_icon="🥇", layout="wide")
sel_lang = st.sidebar.selectbox("🌐 Dil / Язык / Language", ["AZ", "RU", "EN"])
L = LANG[sel_lang]

st.title(f"🥇 {L['title']}")

# --- SIDEBAR: РЕГИОНЫ ---
st.sidebar.header("📍 Location")
reg_choice = st.sidebar.selectbox(L["region"], list(DATA.keys()))
dist_choice = st.sidebar.selectbox(L["district"], DATA[reg_choice]["districts"])
lamps_count = st.sidebar.number_input(L["lamps"], value=DATA[reg_choice]["lamps"])

st.sidebar.divider()
if st.sidebar.button(L["scan_btn"]):
    cap = cv2.VideoCapture(0)
    for _ in range(20): cap.read() # Прогрев камеры
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

# --- РАСЧЕТЫ И ВЫВОД ---
col_vis, col_ctrl = st.columns([1, 2])
with col_vis:
    if st.session_state.frame is not None:
        st.image(st.session_state.frame, channels="BGR", caption=L["objects"])
    else:
        st.info("Scan environment to start AI Vision")

with col_ctrl:
    hour = st.slider(L["time"], 0, 23, 21)
    weather_idx = st.selectbox(L["weather"], [0, 1, 2], format_func=lambda x: L["w_list"][x])
    
    # ПРЕДСКАЗАНИЕ ИИ (Gradient Boosting)
    input_df = pd.DataFrame([[hour, st.session_state.cars, st.session_state.people, weather_idx]], 
                             columns=['hour', 'cars', 'people', 'weather'])
    brightness = int(light_model.predict(input_df)[0]) if light_model else 50
    
    if 6 <= hour <= 18: brightness = 0
    else: brightness = max(min(brightness, 100), 20)

    st.subheader(f"🚗 {st.session_state.cars} | 🚶 {st.session_state.people}")

    # ЭКОНОМИКА
    std_brightness = 100 if (hour < 6 or hour > 18) else 0
    cost_std = (std_brightness/100) * 0.25 * lamps_count * 0.09
    cost_ai = (brightness/100) * 0.25 * lamps_count * 0.09
    savings = cost_std - cost_ai

    m1, m2, m3 = st.columns(3)
    m1.metric(L["metrics"][0], f"{brightness}%")
    m2.metric(L["metrics"][1], f"{cost_ai:.2f} AZN/h")
    m3.metric(L["metrics"][2], f"{savings:.2f} AZN/h", delta=f"{(savings/(cost_std+0.1)*100):.1f}%")

st.plotly_chart(go.Figure([
    go.Bar(name='Standard', x=['Cost'], y=[cost_std], marker_color='#ff4b4b'),
    go.Bar(name='SmartLighting AI', x=['Cost'], y=[cost_ai], marker_color='#00CC96')
]), use_container_width=True)

# --- 🧠 НАСТОЯЩИЙ ИИ ЧАТ (GOOGLE GEMINI) ---
st.divider()
st.subheader(L["chat_head"])

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input(L["chat_in"]):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    
    with st.chat_message("assistant"):
        if has_real_ai:
            with st.spinner("AI Brain thinking..."):
                # ПЕРЕДАЕМ ИИ ВЕСЬ КОНТЕКСТ ПРОЕКТА
                context = f"""
                Sən 'SmartLighting AI' layihəsinin intellektual analitikisən. 
                Sistem Azərbaycan üçün hazırlanıb. 
                Hazırkı vəziyyət:
                - Region: {reg_choice}, Ərazi: {dist_choice}
                - Fənər sayı: {lamps_count}
                - YOLOv8 kamerası tərəfindən aşkar edilib: {st.session_state.cars} maşın və {st.session_state.people} insan.
                - Hazırkı parlaqlıq: {brightness}%.
                - Saatlıq qənaət: {savings:.2f} AZN.
                - Hava: {L['w_list'][weather_idx]}.
                - Metod: Parlaqlıq Gradient Boosting ML modeli tərəfindən müəyyən edilir.
                
                İstifadəçinin sualına ({sel_lang} dilində) bu real məlumatlar əsasında, professional və ağıllı şəkildə cavab ver. 
                """
                try:
                    response = gemini_model.generate_content(context + prompt)
                    full_res = response.text
                except Exception as e:
                    full_res = f"İİ xətası baş verdi. Lütfən API açarını yoxlayın."
        else:
            full_res = "Google Gemini API Key not found. Please add it to secrets.toml"
        
        st.markdown(full_res)
        st.session_state.messages.append({"role": "assistant", "content": full_res})
