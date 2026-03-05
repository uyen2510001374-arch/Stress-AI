import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Tải mô hình đã huấn luyện từ Colab lên
model = joblib.load('stress_model.pkl')

# 2. Giao diện ứng dụng
st.set_page_config(page_title="Stress Predictor", layout="centered")
st.title("🧠 Hệ thống Dự báo Mức độ Căng thẳng")

st.write("Vui lòng nhập các thông số bên dưới để đánh giá trạng thái tâm lý:")

# Tạo 2 cột để giao diện đẹp hơn
col1, col2 = st.columns(2)

with col1:
    anxiety = st.slider("Mức độ lo âu (0-21)", 0, 21, 10)
    depression = st.slider("Mức độ trầm cảm (0-27)", 0, 27, 10)
    sleep = st.slider("Chất lượng giấc ngủ (0-5)", 0, 5, 3)
    headache = st.slider("Tần suất đau đầu (0-5)", 0, 5, 1)

with col2:
    academic = st.slider("Kết quả học tập (0-5)", 0, 5, 3)
    peer_pressure = st.slider("Áp lực đồng lứa (0-5)", 0, 5, 2)
    study_load = st.slider("Khối lượng học tập (0-5)", 0, 5, 2)
    social_support = st.slider("Hỗ trợ xã hội (0-3)", 0, 3, 2)

# Gom nhóm dữ liệu (Sắp xếp đúng thứ tự 20 cột như file CSV gốc)
# Ở đây tôi ví dụ điền các cột còn lại là giá trị trung bình
input_data = [
    anxiety, 20, 0, depression, headache, 2, sleep, 2, 2, 3, 3, 3, 
    academic, study_load, 3, 3, social_support, peer_pressure, 2, 1
]

if st.button("Dự báo kết quả"):
    prediction = model.predict([input_data])[0]
    
    levels = {0: "THẤP ✅", 1: "TRUNG BÌNH ⚠️", 2: "CAO 🚨"}
    
    st.markdown(f"### Kết quả: **{levels[prediction]}**")
    
    if prediction == 2:
        st.error("Cảnh báo: Bạn đang gặp áp lực rất lớn. Hãy tìm kiếm sự trợ giúp từ chuyên gia hoặc người thân.")
    elif prediction == 1:
        st.warning("Ghi chú: Bạn có dấu hiệu căng thẳng. Hãy dành thời gian nghỉ ngơi nhiều hơn.")
    else:
        st.success("Tuyệt vời: Bạn đang kiểm soát tâm lý rất tốt!")