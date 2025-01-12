from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
from collections import Counter

# Khởi tạo Flask app
app = Flask(__name__)

# Load YOLO model
model = YOLO("C:/Users/Admin/Documents/Card/best.pt")  # Thay "best.pt" bằng đường dẫn tới file model của bạn

# Biến toàn cục để lưu nhãn và số lượng
current_labels = {}

# Hàm xử lý video từ webcam
def generate_frames():
    global current_labels
    cap = cv2.VideoCapture(0)  # Mở webcam (camera mặc định)
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Dự đoán với mô hình YOLO
        results = model.predict(source=frame, show=False)

        # Lưu danh sách nhãn phát hiện
        labels = []

        # Vẽ bounding box và tên class lên frame
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Tọa độ bounding box
                conf = box.conf[0]  # Độ chính xác
                cls_id = int(box.cls[0])  # ID class
                label = model.names[cls_id]  # Lấy tên nhãn
                labels.append(label)  # Thêm nhãn vào danh sách

                # Vẽ bounding box và nhãn
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Đếm số lượng nhãn
        current_labels = Counter(labels)

        # Mã hóa frame thành luồng video
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Trả frame về trình duyệt
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Trang chính của ứng dụng
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint stream video từ webcam
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Endpoint gửi danh sách nhãn hiện tại
@app.route('/labels')
def labels():
    return jsonify(current_labels)

# Khởi chạy Flask app
if __name__ == '__main__':
    app.run(debug=True)
