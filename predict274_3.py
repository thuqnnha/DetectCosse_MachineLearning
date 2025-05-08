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
def detect_and_predict(image_path):
    # Tạo thư mục lưu ảnh cắt ra
    if not os.path.exists('cuts'):
        os.makedirs('cuts')

    # Load ảnh
    img = cv2.imread(image_path)
    img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2), interpolation=cv2.INTER_CUBIC)
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
            # Tính toán rotated rectangle và góc
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # Tính góc so với trục hoành (-90° đến 90°)
            angle = rect[2]
            if rect[1][0] < rect[1][1]:  # Nếu width < height
                angle = 90 + angle
                
            if angle > 90:
                angle -= 180
              

            
            # Lấy tọa độ bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Cắt ảnh và dự đoán
            roi = img[y:y+h, x:x+w]
            cv2.imwrite(f"cuts/cut_{i}.jpg", roi)
            label, confidence = predict_image(f"cuts/cut_{i}.jpg")

            # Xác định màu vẽ
            color = (0, 255, 0) if confidence >= 80 else (0, 0, 255)

            # Vẽ rotated rectangle
            cv2.drawContours(img_draw, [box], 0, color, 2)
            
            # Vẽ TRỤC HOÀNH THAM CHIẾU (màu xanh dương)
            center = (int(rect[0][0]), int(rect[0][1]))
            axis_length = max(w, h) * 0.8  # Chiều dài trục
            
            # Trục hoành (màu xanh dương)
            end_point_x = (int(center[0] + axis_length), int(center[1]))
            cv2.line(img_draw, center, end_point_x, (255, 0, 0), 2)
            
            # Đường góc nghiêng (màu vàng)
            end_point_angle = (int(center[0] + axis_length * math.cos(math.radians(angle))),
                             int(center[1] + axis_length * math.sin(math.radians(angle))))
            
            cv2.line(img_draw, center, end_point_angle, (0, 255, 255), 2)
            # Đảo chiều
            angle = -angle
            
            # Hiển thị thông tin
            cv2.putText(img_draw, 
                       f"{label} ({confidence:.1f}%)", 
                       (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       1.5, color, 2)
            
            cv2.putText(img_draw,
                       f"Góc: {angle:.1f}°",
                       (x, y+h+20),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       1.5, (255, 255, 255), 2)

    # Hiển thị ảnh kết quả
    result = cv2.resize(img_draw, None, fx=0.3, fy=0.3)
    cv2.imshow("Kết quả nhận diện", result)
    cv2.imwrite("result.jpg", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_and_predict("image_input.jpg")
