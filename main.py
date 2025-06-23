from ultralytics import YOLO
import os
import numpy as np

# ==== Ghi kết quả dự đoán ra file .txt theo định dạng YOLO ====
def save_yolo_txt(result, save_path):
    with open(save_path, 'w') as f:
        for box in result.boxes:
            cls_id = int(box.cls)
            xywh = box.xywh[0].cpu().numpy()
            img_w, img_h = result.orig_shape[1], result.orig_shape[0]
            norm_xywh = xywh / np.array([img_w, img_h, img_w, img_h])
            norm_xywh = np.clip(norm_xywh, 0, 1)
            f.write(f"{cls_id} {norm_xywh[0]:.6f} {norm_xywh[1]:.6f} {norm_xywh[2]:.6f} {norm_xywh[3]:.6f}\n")

# ==== 1. Huấn luyện mô hình từ đầu ====
def train_model():
    print(" Bắt đầu huấn luyện mô hình YOLOv8...")
    model = YOLO('yolov8n.pt')
    model.train(
        data='data.yaml',
        epochs=5,                 #Chạy demo kiểm tra nên chỉ dùng 5 epochs
        imgsz=640,
        batch=16,
        name='car_phone_model',
        project='runs/train',
        save_period=1            #Luôn lưu mô hình mỗi epoch
    )
    print(" Huấn luyện hoàn tất!")

# ==== 2. Dự đoán và thống kê hành vi lái xe ====
def predict_and_analyze():
    model_path = "runs/detect/car_phone_model/weights/best.pt"
    test_img_folder = "dataset/images/test"
    pred_label_folder = "predictions/labels/test"
    statistics_file = "predictions/statistics.txt"
    os.makedirs(pred_label_folder, exist_ok=True)

    model = YOLO(model_path)

    count_vi_pham = 0
    count_an_toan = 0
    count_khong_ro = 0

    print(" ĐANG DỰ ĐOÁN VÀ PHÂN TÍCH...")

    for filename in os.listdir(test_img_folder):
        if filename.lower().endswith(('.jpg', '.png')):
            img_path = os.path.join(test_img_folder, filename)
            results = model(img_path)
            result = results[0]

            detections = result.boxes.cls.cpu().numpy().tolist()
            has_wheel = 0 in detections
            has_phone = 1 in detections

            base_name = os.path.splitext(filename)[0]
            save_path = os.path.join(pred_label_folder, base_name + ".txt")
            save_yolo_txt(result, save_path)

            if has_wheel and has_phone:
                count_vi_pham += 1
            elif has_wheel and not has_phone:
                count_an_toan += 1
            else:
                count_khong_ro += 1

    total = count_vi_pham + count_an_toan + count_khong_ro
    stat_text = (
        "=== THỐNG KÊ HÀNH VI LÁI XE ===\n"
        f" VI PHẠM    : {count_vi_pham} ảnh\n"
        f" AN TOÀN    : {count_an_toan} ảnh\n"
        f" KHÔNG RÕ   : {count_khong_ro} ảnh\n"
        f" TỔNG SỐ ẢNH: {total} ảnh\n"
    )

    print("\n" + stat_text)
    with open(statistics_file, "w", encoding="utf-8") as f:
        f.write(stat_text)

    print(f" Đã lưu thống kê tại: {statistics_file}")
    print(" Dự đoán và thống kê hoàn tất!")

# ==== 3. Đánh giá độ chính xác mô hình ====
def evaluate_model():
    print(" Đang đánh giá độ chính xác của mô hình trên tập validation...")

    model = YOLO("runs/detect/car_phone_model/weights/best.pt")

    val_results = model.val(
        data="data.yaml",
        split="val",
        save=True,
        verbose=False
    )

    precision = val_results.box.mp
    recall = val_results.box.mr
    map50 = val_results.box.map50
    map5095 = val_results.box.map

    print("\n KẾT QUẢ ĐÁNH GIÁ:")
    print(f"Precision      : {precision:.3f}")
    print(f"Recall         : {recall:.3f}")
    print(f"mAP@0.5        : {map50:.3f}")
    print(f"mAP@0.5:0.95   : {map5095:.3f}")
    print(" Đã lưu biểu đồ và báo cáo tại: runs/detect/")

# ==== 4. Hàm main chọn chế độ ====
def main():
    mode = "eval"  #Chọn 1 trong 3 chế độ: "train", "predict", "eval"

    if mode == "train":
        train_model()
    elif mode == "predict":
        predict_and_analyze()
    elif mode == "eval":
        evaluate_model()
    else:
        print(" Chế độ không hợp lệ. Chọn 'train', 'predict' hoặc 'eval'.")

# ==== 5. Gọi chương trình chính ====
if __name__ == "__main__":
    main()
