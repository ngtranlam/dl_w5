import streamlit as st
import joblib
import requests
from bs4 import BeautifulSoup
import plotly.express as px

# Đọc mô hình Naive Bayes và vectơ hóa TF-IDF từ các tệp
nb_classifier = joblib.load('nb_classifier.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Thêm thông tin cá nhân vào thanh sidebar
st.sidebar.markdown("## Họ tên: Nguyễn Trần Lâm")
st.sidebar.markdown("## MSSV: 20016701")
st.sidebar.markdown("## BÀI TẬP DEEPLEARNING TUẦN 5")

# Đặt màu nền cho toàn bộ ứng dụng
st.markdown("""
    <style>
    body {
        background-color: #f0f0f0;
        font-family: Arial, sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# Đặt màu cho tiêu đề
st.markdown('<h1 style="color:#0066cc;">Phân loại Văn bản</h1>', unsafe_allow_html=True)

# Lựa chọn nhập dữ liệu
data_option = st.radio("Chọn cách nhập dữ liệu", ("Nhập trực tiếp văn bản", "Tải lên file", "Lấy văn bản từ link bài báo"))

if data_option == "Nhập trực tiếp văn bản":
    # Thêm textarea để nhập văn bản
    text = st.text_area("Nhập văn bản cần phân loại", "")

    # Thêm nút "Phân loại"
    classify_button = st.button("Phân loại")

    # Kiểm tra xem nút "Phân loại" đã được nhấn hay chưa
    if classify_button:
        if text:
            # Biểu diễn văn bản bằng vectơ TF-IDF và chuyển đổi thành ma trận mật độ
            text_vector = tfidf_vectorizer.transform([text]).toarray()

            # Dự đoán lớp của văn bản bằng mô hình Naive Bayes
            predicted_class_number = nb_classifier.predict(text_vector)

            # Ánh xạ mã số thành tên loại văn bản
            label_mapping = {0: "Doanh nghiệp", 1: "Chính trị", 2: "Thể thao"}
            predicted_class_name = label_mapping.get(predicted_class_number[0], "Không xác định")

            # Hiển thị kết quả phân loại
            st.markdown(f'<div style="font-size:20px; color:#009900;">Kết quả phân loại: {predicted_class_name}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-size:16px; color:#ff0000;">Vui lòng nhập văn bản trước khi phân loại.</div>', unsafe_allow_html=True)

elif data_option == "Tải lên file":
    # Thêm lựa chọn tải lên file dữ liệu
    uploaded_file = st.file_uploader("Tải lên file dữ liệu (chỉ hỗ trợ file .txt)", type=["txt"])

    # Thêm nút "Phân loại từ file"
    classify_file_button = st.button("Phân loại từ file")

    # Kiểm tra xem nút "Phân loại từ file" đã được nhấn hay chưa
    if classify_file_button:
        if uploaded_file is not None:
            # Đọc nội dung của file
            data = uploaded_file.read()

            # Biểu diễn nội dung file bằng vectơ TF-IDF và chuyển đổi thành ma trận mật độ
            data_vector = tfidf_vectorizer.transform([data]).toarray()

            # Dự đoán lớp của dữ liệu bằng mô hình Naive Bayes
            predicted_class_number = nb_classifier.predict(data_vector)

            # Ánh xạ mã số thành tên loại văn bản
            label_mapping = {0: "Doanh nghiệp", 1: "Chính trị", 2: "Thể thao"}
            predicted_class_name = label_mapping.get(predicted_class_number[0], "Không xác định")

            # Hiển thị kết quả phân loại cho dữ liệu từ file
            st.markdown(f'<div style="font-size:20px; color:#009900;">Kết quả phân loại từ file: {predicted_class_name}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-size:16px; color:#ff0000;">Vui lòng tải lên file dữ liệu trước khi phân loại từ file.</div>', unsafe_allow_html=True)

elif data_option == "Lấy văn bản từ link bài báo":
    # Thêm lựa chọn nhập URL
    url = st.text_input("Nhập URL của bài báo", "")

    # Thêm nút "Phân loại từ URL"
    classify_url_button = st.button("Phân loại từ URL")

    # Kiểm tra xem nút "Phân loại từ URL" đã được nhấn hay chưa
    if classify_url_button:
        if url:
            try:
                # Truy cập URL và lấy nội dung
                response = requests.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                article_text = ' '.join([p.text for p in soup.find_all('p')])

                # Biểu diễn nội dung bài báo bằng vectơ TF-IDF và chuyển đổi thành ma trận mật độ
                article_vector = tfidf_vectorizer.transform([article_text]).toarray()

                # Dự đoán lớp của nội dung bài báo bằng mô hình Naive Bayes
                predicted_class_number = nb_classifier.predict(article_vector)

                # Ánh xạ mã số thành tên loại văn bản
                label_mapping = {0: "Doanh nghiệp", 1: "Chính trị", 2: "Thể thao"}
                predicted_class_name = label_mapping.get(predicted_class_number[0], "Không xác định")

                # Hiển thị kết quả phân loại từ URL
                st.markdown(f'<div style="font-size:20px; color:#009900;">Kết quả phân loại từ URL: {predicted_class_name}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f'<div style="font-size:16px; color:#ff0000;">Lỗi: {str(e)}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-size:16px; color:#ff0000;">Vui lòng nhập URL của bài báo trước khi phân loại từ URL.</div>', unsafe_allow_html=True)
