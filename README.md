# BTL_XLA_Nhom1
Báo cáo BTL môn Xử lý ảnh số và thị giác máy tính
Nhóm 1_Lớp 20242IT6072002_Khóa K17
Đề tài: Xây dựng hệ thống nhận diện người đang sử dụng điện thoại khi lái xe
Mục tiêu: Xây dựng được mô hình nhận diện người đang sử dụng điện thoại khi lái xe bằng YOLOv8
Nội dung tài liệu gồm:
  1. File main.py: chứa toàn bộ code chương trình gồm:
     + các thư viện cần thiết
     + hàm ghi kết quả dự đoán ra file .txt theo định dạng YOLO: save_yolo_txt(result, save_path)
     + hàm huấn luyện mô hình từ đầu train_model()
     + hàm dự đoán và thống kê hành vi lái xe predict_and_analyze()
     + hàm đánh giá độ chính xác của mô hình được huấn luyện evaluate_model()
     + hàm main() chọn 1 trong 3 chế độ của chương trình
       Chế độ 1: mode = "train"   : Huấn luyện mô hình
       Chế độ 2: mode = "predict" : Dự đoán và thống kê
       Chế độ 3: mode = "eval"    : Đánh giá độ chính xác mô hình được huấn luyện
  2. File data.yaml: Mô tả cách tổ chức dữ liệu huấn luyện, giúp cung cấp thông tin cho mô hình:
                     + Vị trí tập dữ liệu
                     + Số lượng nhãn
                     + Tên từng nhãn
  3. runs/detect/car_phone_model/weights/best.pt
     + File best.pt này là file mô hình được huấn luyện tốt nhất được huấn luyện trước, thuận tiện cho việc sử dụng mode predict và eval.
     + File được tạo bằng cách dùng hàm huấn luyện trong main.py để tạo mô hình huấn luyện trên Google Colab (Sử dụng cùng một bộ dữ liệu huấn luyện với chương trình dùng cho chương trình main.py, dùng GPU cho tốc độ huấn luyện nhanh hơn, epochs=50 để mô hình được huấn luyện tốt nhất)
