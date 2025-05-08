import cv2
import numpy as np
import tensorflow as tf
import os
import math

# --- Cấu hình ---
img_height = 180
img_width = 180

# --- Load model đã train ---
model = tf.keras.models.load_model("model_cos.h5")

# --- Load class names từ dataset ---
dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    image_size=(img_height, img_width),
    batch_size=1
)
class_names = dataset.class_names

# --- Hàm predict 1 ảnh nhỏ ---
def predict_image(image_path):
    img = tf.keras.utils.load_img(
        image_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array, verbose=0)
    score = tf.nn.softmax(predictions[0])
    return class_names[np.argmax(score)], 100 * np.max(score)

# --- Hàm xử lý và cắt ảnh ---
def detect_and_predict_video(video_path):
    # Tạo thư mục lưu ảnh cắt ra
    if not os.path.exists('cuts'):
        os.makedirs('cuts')

    # Mở video
    cap = cv2.VideoCapture(video_path)

    # Kiểm tra nếu video được mở thành công
    if not cap.isOpened():
        
        print("Không thể mở video!")
        return

    object_id = 1  # Khởi tạo ID bắt đầu từ 1
    tracked_objects = {}  # ID: (center_x, center_y)
    
    frame_count = 0  # Thêm ở trên trước vòng while
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        skip_frame = 5  # Xử lý mỗi 5 frame, chỉnh theo ý bạn
        if frame_count % skip_frame != 0:
            continue

        # Tăng kích thước video
        img = cv2.resize(frame, (frame.shape[1]*2, frame.shape[0]*2), interpolation=cv2.INTER_CUBIC)
        img_draw = img.copy()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Tạo mask màu
        masks = []
        lower_blue = np.array([95, 120, 120])
        upper_blue = np.array([125, 255, 255])
        masks.append(cv2.inRange(hsv, lower_blue, upper_blue))
        
        lower_red1 = np.array([0, 80, 80])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 80, 80])
        upper_red2 = np.array([179, 255, 255])
        masks.append(cv2.inRange(hsv, lower_red1, upper_red1))
        masks.append(cv2.inRange(hsv, lower_red2, upper_red2))
        
        lower_yellow = np.array([15, 70, 70])
        upper_yellow = np.array([40, 255, 255])
        masks.append(cv2.inRange(hsv, lower_yellow, upper_yellow))

        # Gộp mask và lọc nhiễu
        full_mask = masks[0]
        for m in masks[1:]:
            full_mask = cv2.bitwise_or(full_mask, m)
        
        kernel = np.ones((3, 3), np.uint8)
        full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel)
        full_mask = cv2.dilate(full_mask, kernel, iterations=1)

        # Tìm contours
        contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"Phát hiện {len(contours)} đầu cos")

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 300:
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                angle = rect[2]
                if rect[1][0] < rect[1][1]:
                    angle = 90 + angle
                if angle > 90:
                    angle -= 180
                x, y, w, h = cv2.boundingRect(contour)
                roi = img[y:y+h, x:x+w]
                cut_path = f"cuts/cut_{i}.jpg"
                cv2.imwrite(cut_path, roi)
                label, confidence = predict_image(cut_path)

                if confidence >= 95:
                    center_x = int(rect[0][0])
                    center_y = int(rect[0][1])
                    assigned_id = None

                    # So sánh với các object đã detect trước
                    for id, (prev_x, prev_y) in tracked_objects.items():
                        distance = math.hypot(center_x - prev_x, center_y - prev_y)
                        if distance < 100:
                            assigned_id = id
                            break

                    # Nếu không trùng khớp với object nào trước, gán ID mới
                    if assigned_id is None:
                        assigned_id = object_id
                        tracked_objects[assigned_id] = (center_x, center_y)
                        object_id += 1

                    # Vẽ rotated rectangle
                    color = (0, 255, 0)
                    cv2.drawContours(img_draw, [box], 0, color, 2)

                    # Vẽ trục tham chiếu
                    axis_length = max(w, h) * 0.8
                    end_point_x = (int(center_x + axis_length), int(center_y))
                    cv2.line(img_draw, (center_x, center_y), end_point_x, (255, 0, 0), 2)

                    end_point_angle = (int(center_x + axis_length * math.cos(math.radians(angle))),
                                       int(center_y + axis_length * math.sin(math.radians(angle))))
                    cv2.line(img_draw, (center_x, center_y), end_point_angle, (0, 255, 255), 2)

                    angle = -angle  # Đảo chiều

                    # Hiển thị các thông tin
                    cv2.putText(img_draw,
                                f"ID: {assigned_id}, Toa do: ({x}, {y})",
                               (x, y-30),
                               cv2.FONT_HERSHEY_SIMPLEX,
                               1.5, (0, 0, 0), 2)

                    cv2.putText(img_draw,
                                f"Goc: {angle:.1f}°",
                               (x, y+h+40),
                               cv2.FONT_HERSHEY_SIMPLEX,
                               1.5, (0, 0, 0), 2)

                    cv2.putText(img_draw,
                                f"Dien tich: {area:.1f} px",
                               (x, y+h+100),
                               cv2.FONT_HERSHEY_SIMPLEX,
                               1.5, (0, 0, 0), 2)

                    cv2.putText(img_draw,
                                f"{label} ({confidence:.1f}%)",
                               (x, y+h+170),
                               cv2.FONT_HERSHEY_SIMPLEX,
                               1.5, (0, 0, 0), 2)

                    cv2.putText(img_draw,
                                "Vat the phat hien co do chinh xac >=95%, nhan Enter de tiep tuc hoac 'q' de thoat",
                               (50, 50),
                               cv2.FONT_HERSHEY_SIMPLEX,
                               0.8, (0, 0, 255), 2)

                    result = cv2.resize(img_draw, None, fx=0.3, fy=0.3)
                    cv2.imshow("Ket qua nhan dien", result)

                    while True:
                        key = cv2.waitKey(0)
                        if key == 13:
                            break
                        elif key == ord('q'):
                            print("Thoat chuong trinh...")
                            cap.release()
                            cv2.destroyAllWindows()
                            return

      
        result = cv2.resize(img_draw, None, fx=0.3, fy=0.3)
        cv2.imshow("Ket qua nhan dien", result)

        while True:
            key = cv2.waitKey(0)
            if key == 13:
                break
            elif key == ord('q'):
                print("Thoat chuong trinh...")
                cap.release()
                cv2.destroyAllWindows()
                return

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_and_predict_video("video_input.mp4")
