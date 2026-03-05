import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Cấu hình trang và Giao diện
st.set_page_config(page_title="Stress Predictor", layout="wide")
st.title("🧠 Hệ thống Đánh giá & Dự báo Căng thẳng (Stress)")
st.markdown("---")

# 2. Tải mô hình đã huấn luyện (Đảm bảo file .pkl nằm cùng thư mục trên Github)
@st.cache_resource
def load_model():
    return joblib.load('stress_model.pkl')

model = load_model()

# 3. Tạo Form nhập liệu với chú giải chi tiết
st.subheader("📊 Hãy kéo các thanh trượt bên dưới để khớp với trạng thái của bạn:")

with st.form("stress_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.write("#### 🧘 Sức khỏe Tâm thần")
        anxiety = st.slider("Mức độ Lo âu (Anxiety Level)", 0, 21, 10, 
                           help="0: Bình tĩnh/Thoải mái | 21: Rất lo âu/Hoảng loạn")
        st.caption("*(0: Cực kỳ tốt - 21: Cực kỳ tệ)*")

        depression = st.slider("Mức độ Trầm cảm (Depression)", 0, 27, 10,
                              help="0: Vui vẻ/Lạc quan | 27: U uất/Thất vọng nặng")
        st.caption("*(0: Cực kỳ tốt - 27: Cực kỳ tệ)*")

        sleep = st.slider("Chất lượng Giấc ngủ (Sleep Quality)", 0, 5, 3)
        st.caption("*(0: Rất tệ/Mất ngủ - 5: Rất tốt/Ngủ sâu)*")

        headache = st.slider("Tần suất Đau đầu (Headache)", 0, 5, 1)
        st.caption("*(0: Không bao giờ - 5: Đau liên tục hàng ngày)*")

    with col2:
        st.write("#### 🏫 Học tập & Xã hội")
        academic = st.slider("Kết quả học tập (Academic Performance)", 0, 5, 3)
        st.caption("*(0: Rất kém/Yếu - 5: Xuất sắc/Giỏi)*")

        study_load = st.slider("Khối lượng học tập (Study Load)", 0, 5, 2)
        st.caption("*(0: Rất nhẹ nhàng - 5: Quá tải/Áp lực cực lớn)*")

        peer_pressure = st.slider("Áp lực từ bạn bè (Peer Pressure)", 0, 5, 2)
        st.caption("*(0: Không áp lực - 5: Bị so sánh/Áp lực rất nặng)*")

        social_support = st.slider("Hỗ trợ xã hội (Social Support)", 0, 3, 2)
        st.caption("*(0: Cô đơn/Không ai giúp - 3: Có gia đình/Bạn bè ủng hộ nhiều)*")

    # Nút bấm tính toán
    submitted = st.form_submit_button("PHÂN TÍCH KẾT QUẢ")

# 4. Xử lý logic dự báo
if submitted:
    # Thứ tự các cột phải khớp 100% với lúc huấn luyện trên Colab (20 cột)
    # Các cột không có thanh trượt sẽ được lấy giá trị trung bình giả định (2-3)
    # [anxiety, self_esteem, mental_health_history, depression, headache, blood_pressure, sleep_quality, 
    # breathing_problem, noise_level, living_conditions, safety, basic_needs, academic_performance, 
    # study_load, teacher_student_relationship, future_career_concerns, social_support, peer_pressure, 
    # extracurricular_activities, bullying]
    
    features = [
        anxiety, 20, 0, depression, headache, 2, sleep, 
        2, 2, 3, 3, 3, academic, study_load, 3, 3, 
        social_support, peer_pressure, 2, 1
    ]

    prediction = model.predict([features])[0]
    
    st.markdown("---")
    st.subheader("📌 Kết quả chẩn đoán:")
    
    if prediction == 2:
        st.error("### Mức độ Stress: CAO (🚨)")
        st.markdown("""
        **Lời khuyên:** 
        - Bạn đang ở mức báo động. Hãy tạm dừng công việc/học tập ngay lập tức.
        - Tìm kiếm sự trợ giúp từ bác sĩ tâm lý hoặc người thân tín nhất.
        - Thực hiện các bài tập thở sâu và nghỉ ngơi ít nhất 7-8 tiếng mỗi ngày.
        """)
    elif prediction == 1:
        st.warning("### Mức độ Stress: TRUNG BÌNH (⚠️)")
        st.markdown("""
        **Lời khuyên:** 
        - Bạn bắt đầu có dấu hiệu quá tải. 
        - Hãy sắp xếp lại thời gian biểu, giảm bớt khối lượng học tập.
        - Dành thời gian cho sở thích cá nhân để cân bằng lại cảm xúc.
        """)
    else:
        st.success("### Mức độ Stress: THẤP (✅)")
        st.markdown("""
        **Lời khuyên:** 
        - Chúc mừng! Bạn đang kiểm soát cuộc sống rất tốt. 
        - Tiếp tục duy trì lối sống lành mạnh và chế độ ngủ nghỉ hiện tại.
        """)

# Thêm chân trang
st.markdown("---")
st.caption("Lưu ý: Công cụ này chỉ mang tính chất tham khảo dựa trên dữ liệu máy học, không thay thế cho chẩn đoán y khoa chuyên nghiệp.")
