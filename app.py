import streamlit as st
import pandas as pd
import joblib

# 1. Cấu hình trang
st.set_page_config(page_title="Stress Predictor", layout="wide")
st.title("🧠 HỆ THỐNG DỰ BÁO MỨC ĐỘ CĂNG THẲNG")
st.markdown("---")

# 2. Tải mô hình
@st.cache_resource
def load_model():
    return joblib.load('stress_model.pkl')

model = load_model()

# 3. Giao diện nhập liệu
st.subheader("📝 Vui lòng nhập các chỉ số của bạn (Lưu ý các mốc đánh giá):")

with st.form("stress_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🧘 Sức khỏe & Tâm lý")
        
        # Ghi rõ mốc 0 và mốc tối đa ngay trong nhãn
        anxiety = st.slider("1. Mức độ LO ÂU (0: Thấp/Bình thản --- 21: Rất cao/Hoảng loạn)", 0, 21, 10)
        
        depression = st.slider("2. Mức độ TRẦM CẢM (0: Thấp/Vui vẻ --- 27: Rất cao/U uất)", 0, 27, 10)
        
        sleep = st.slider("3. Chất lượng GIẤC NGỦ (0: Cực kỳ tệ/Mất ngủ --- 5: Cực kỳ tốt/Ngủ sâu)", 0, 5, 3)
        
        headache = st.slider("4. Tần suất ĐAU ĐẦU (0: Không bao giờ --- 5: Đau rất thường xuyên)", 0, 5, 1)

    with col2:
        st.markdown("### 🏫 Học tập & Xã hội")
        
        academic = st.slider("5. Kết quả HỌC TẬP (0: Rất kém/Yếu --- 5: Rất tốt/Giỏi)", 0, 5, 3)
        
        study_load = st.slider("6. Khối lượng HỌC TẬP (0: Rất nhẹ nhàng --- 5: Quá tải/Áp lực)", 0, 5, 2)
        
        peer_pressure = st.slider("7. Áp lực BẠN BÈ (0: Không có áp lực --- 5: Áp lực rất lớn)", 0, 5, 2)
        
        social_support = st.slider("8. Hỗ trợ XÃ HỘI (0: Không có ai giúp đỡ --- 3: Được hỗ trợ tối đa)", 0, 3, 2)

    st.markdown("---")
    submitted = st.form_submit_button("🚀 NHẤN VÀO ĐÂY ĐỂ PHÂN TÍCH KẾT QUẢ")

# 4. Xử lý kết quả
if submitted:
    # Danh sách tên cột chính xác như khi huấn luyện để tránh lỗi cảnh báo
    columns = [
        'anxiety_level', 'self_esteem', 'mental_health_history', 'depression',
        'headache', 'blood_pressure', 'sleep_quality', 'breathing_problem',
        'noise_level', 'living_conditions', 'safety', 'basic_needs',
        'academic_performance', 'study_load', 'teacher_student_relationship',
        'future_career_concerns', 'social_support', 'peer_pressure',
        'extracurricular_activities', 'bullying'
    ]
    
    # Chuẩn bị dữ liệu (Điền các cột còn thiếu bằng giá trị trung bình hợp lý)
    features = [
        anxiety, 20, 0, depression, headache, 2, sleep, 
        2, 2, 3, 3, 3, academic, study_load, 3, 3, 
        social_support, peer_pressure, 2, 1
    ]
    
    # Chuyển thành DataFrame để mất lỗi "Feature Names"
    input_df = pd.DataFrame([features], columns=columns)
    
    # Dự báo
    prediction = model.predict(input_df)[0]
    
    # Hiển thị kết quả
    if prediction == 2:
        st.error("## KẾT QUẢ: MỨC ĐỘ CĂNG THẲNG CAO (🚨)")
        st.info("💡 Lời khuyên: Bạn nên tìm người chia sẻ hoặc gặp chuyên gia tư vấn để giải tỏa áp lực.")
    elif prediction == 1:
        st.warning("## KẾT QUẢ: MỨC ĐỘ CĂNG THẲNG TRUNG BÌNH (⚠️)")
        st.info("💡 Lời khuyên: Hãy dành thời gian nghỉ ngơi, tập thể dục và cân bằng lại việc học tập.")
    else:
        st.success("## KẾT QUẢ: MỨC ĐỘ CĂNG THẲNG THẤP (✅)")
        st.info("💡 Lời khuyên: Trạng thái của bạn rất tốt. Hãy tiếp tục duy trì lối sống tích cực này!")
