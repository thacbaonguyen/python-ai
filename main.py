import cv2
import mediapipe as mp
import pygame
import numpy as np
import os
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import logging
import joblib

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Khởi tạo các module MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Khởi tạo mixer pygame để phát âm thanh
try:
    pygame.mixer.init()
    logger.info("Khởi tạo pygame mixer thành công")
except Exception as e:
    logger.error(f"Lỗi khi khởi tạo pygame mixer: {e}")
    exit(1)

# Định nghĩa thư mục lưu mô hình và dữ liệu
MODEL_DIR = 'models'
DATA_DIR = 'data'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Load các file âm thanh vào một dictionary
sounds = {}
sound_files = {
    "fa_sharp": "fa_sharp.wav",  # Ngón trỏ tay trái
    "a": "a.wav",                # Ngón giữa tay trái
    "re": "re.wav",              # Ngón áp út tay trái
    "do_sharp": "do_sharp.wav",  # Ngón trỏ tay phải
    "sol_sharp": "sol_sharp.wav",# Ngón giữa tay phải
    "si": "si.wav"               # Ngón áp út tay phải
}

# Tải các file âm thanh
try:
    for name, file in sound_files.items():
        file_path = os.path.join(DATA_DIR, file)
        if os.path.exists(file_path):
            sounds[name] = pygame.mixer.Sound(file_path)
        else:
            logger.warning(f"File âm thanh {file_path} không tồn tại")
    logger.info("Đã tải các file âm thanh")
except Exception as e:
    logger.error(f"Lỗi khi tải file âm thanh: {e}")

# Ánh xạ giữa ngón tay và âm thanh
finger_to_sound = {
    "left_index": "fa_sharp",
    "left_middle": "a",
    "left_ring": "re",
    "right_index": "do_sharp",
    "right_middle": "sol_sharp",
    "right_ring": "si"
}

# Các điểm landmark của từng ngón tay
FINGER_TIPS = {
    "index": 8,
    "middle": 12,
    "ring": 16
}

FINGER_MCP = {
    "index": 5,
    "middle": 9,
    "ring": 13
}

# Số lượng features cho mô hình
NUM_FEATURES = 63  # 21 landmarks x 3 dimensions (x,y,z)

# Hàm trích xuất đặc trưng từ landmark của bàn tay
def extract_features(hand_landmarks):
    features = []
    for landmark in hand_landmarks.landmark:
        features.extend([landmark.x, landmark.y, landmark.z])
    return features

# Hàm trích xuất đặc trưng cho cử chỉ động
def extract_dynamic_features(current_landmarks, previous_landmarks):
    features = []
    if previous_landmarks:
        for i, landmark in enumerate(current_landmarks.landmark):
            prev_landmark = previous_landmarks.landmark[i]
            # Tính vận tốc (thay đổi vị trí)
            velocity_x = landmark.x - prev_landmark.x
            velocity_y = landmark.y - prev_landmark.y
            velocity_z = landmark.z - prev_landmark.z
            features.extend([velocity_x, velocity_y, velocity_z])
    else:
        # Nếu không có dữ liệu trước đó, dùng 0 làm giá trị mặc định
        features = [0] * (21 * 3)
    return features

# Hàm kiểm tra xem một ngón tay có được hạ xuống không
def is_finger_down(landmarks, finger_tip, finger_mcp):
    return landmarks[finger_tip].y > landmarks[finger_mcp].y

# Hàm để phát hiện và phân loại cử chỉ tay
class HandGestureRecognizer:
    def __init__(self):
        # Mô hình Random Forest cho phân loại cử chỉ
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Mô hình K-means cho phân cụm cử chỉ
        self.kmeans = KMeans(n_clusters=6, random_state=42)
        
        # Bộ chuẩn hóa dữ liệu
        self.scaler = StandardScaler()
        
        # Dữ liệu huấn luyện
        self.X = []
        self.y = []
        
        # Dữ liệu cho phân cụm
        self.cluster_data = []
        
        # Đường dẫn file lưu mô hình
        self.model_path = os.path.join(MODEL_DIR, "hand_gesture_model.pkl")
        self.kmeans_path = os.path.join(MODEL_DIR, "kmeans_model.pkl")
        self.scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
        
        # Tải mô hình nếu đã tồn tại
        self.load_models()
    
    def collect_data(self, features, label):
        """Thu thập dữ liệu huấn luyện"""
        self.X.append(features)
        self.y.append(label)
        self.cluster_data.append(features)
    
    def train(self):
        """Huấn luyện mô hình Random Forest"""
        if len(self.X) < 10 or len(set(self.y)) < 2:
            logger.warning("Không đủ dữ liệu để huấn luyện mô hình")
            return False
        
        try:
            # Chuẩn hóa dữ liệu
            X_scaled = self.scaler.fit_transform(self.X)
            
            # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, self.y, test_size=0.2, random_state=42)
            
            # Huấn luyện mô hình
            self.model.fit(X_train, y_train)
            
            # Đánh giá mô hình
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Độ chính xác mô hình: {accuracy}")
            
            # Lưu mô hình
            self.save_models()
            
            return True
        except Exception as e:
            logger.error(f"Lỗi khi huấn luyện mô hình: {e}")
            return False
    
    def train_kmeans(self):
        """Huấn luyện mô hình K-means cho phân cụm cử chỉ"""
        if len(self.cluster_data) < 10:
            logger.warning("Không đủ dữ liệu để huấn luyện mô hình K-means")
            return False
        
        try:
            # Chuẩn hóa dữ liệu
            data_scaled = self.scaler.transform(self.cluster_data)
            
            # Huấn luyện mô hình K-means
            self.kmeans.fit(data_scaled)
            
            # Lưu mô hình
            self.save_models()
            
            return True
        except Exception as e:
            logger.error(f"Lỗi khi huấn luyện mô hình K-means: {e}")
            return False
    
    def predict(self, features):
        """Dự đoán cử chỉ sử dụng mô hình Random Forest"""
        try:
            # Chuẩn hóa đặc trưng
            features_scaled = self.scaler.transform([features])
            
            # Dự đoán
            prediction = self.model.predict(features_scaled)
            return prediction[0]
        except Exception as e:
            logger.error(f"Lỗi khi dự đoán cử chỉ: {e}")
            return None
    
    def cluster(self, features):
        """Phân cụm cử chỉ sử dụng mô hình K-means"""
        try:
            # Chuẩn hóa đặc trưng
            features_scaled = self.scaler.transform([features])
            
            # Phân cụm
            cluster = self.kmeans.predict(features_scaled)
            return cluster[0]
        except Exception as e:
            logger.error(f"Lỗi khi phân cụm cử chỉ: {e}")
            return None
    
    def save_models(self):
        """Lưu mô hình vào file"""
        try:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.kmeans, self.kmeans_path)
            joblib.dump(self.scaler, self.scaler_path)
            logger.info("Đã lưu các mô hình")
        except Exception as e:
            logger.error(f"Lỗi khi lưu mô hình: {e}")
    
    def load_models(self):
        """Tải mô hình từ file"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logger.info("Đã tải mô hình Random Forest")
            
            if os.path.exists(self.kmeans_path):
                self.kmeans = joblib.load(self.kmeans_path)
                logger.info("Đã tải mô hình K-means")
            
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                logger.info("Đã tải bộ chuẩn hóa dữ liệu")
                
            return True
        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình: {e}")
            return False

# Hàm chính
def main():
    # Khởi tạo recognizer
    recognizer = HandGestureRecognizer()
    
    # Video source - Sử dụng webcam mặc định
    video_source = 0  # 0 cho webcam mặc định, 1 cho webcam phụ
    logger.info(f"Sử dụng camera với ID: {video_source}")
    
    # Khởi tạo video capture
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        logger.error(f"Không thể mở camera với ID: {video_source}")
        # Thử với camera khác nếu camera mặc định không hoạt động
        alternative_source = 1
        logger.info(f"Thử với camera phụ ID: {alternative_source}")
        cap = cv2.VideoCapture(alternative_source)
        if not cap.isOpened():
            logger.error(f"Không thể mở bất kỳ camera nào")
            return
    
    # Độ phân giải video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Độ phân giải camera: {width}x{height}")
    
    # Trạng thái của các ngón tay
    finger_state = {
        "left_index": False,
        "left_middle": False, 
        "left_ring": False,
        "right_index": False,
        "right_middle": False,
        "right_ring": False
    }
    
    # Thời gian cuối cùng phát âm thanh cho mỗi ngón
    last_played = {finger: 0 for finger in finger_state.keys()}
    cooldown = 0.5  # Thời gian chờ giữa các lần phát (giây)
    
    # Biến để lưu trữ landmarks của frame trước đó
    previous_landmarks = {
        "Left": None,
        "Right": None
    }
    
    # Biến để thu thập dữ liệu cho việc huấn luyện mô hình
    collect_data = False
    current_gesture_label = None
    
    # Lưu lịch sử vị trí để phân tích chuyển động
    position_history = []
    max_history = 100
    
    # Khởi tạo thuật toán DTW (Dynamic Time Warping) để so khớp chuỗi cử chỉ
    gesture_templates = {}
    current_recording = []
    is_recording = False
    
    # Tạo cửa sổ hiển thị
    window_name = "Chơi nhạc bằng cử chỉ ngón tay"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Khởi tạo MediaPipe Hands
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:
        
        logger.info("Bắt đầu xử lý video từ camera...")
        while cap.isOpened():
            # Đọc frame từ camera
            ret, frame = cap.read()
            if not ret:
                logger.error("Không thể đọc frame từ camera.")
                break
            
            # Lật ngang frame để hiển thị như gương
            frame = cv2.flip(frame, 1)
            
            # Chuyển đổi từ BGR sang RGB (MediaPipe yêu cầu định dạng RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Xử lý frame để phát hiện các bàn tay
            results = hands.process(rgb_frame)
            
            # Vẽ hướng dẫn sử dụng lên frame
            cv2.putText(frame, "Press R: Record gesture | T: Train model | P: Prediction mode", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Hiển thị trạng thái ghi âm nếu đang ghi
            if is_recording:
                cv2.putText(frame, "Recording Gesture...", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Kiểm tra nếu có phát hiện bàn tay
            if results.multi_hand_landmarks:
                hand_landmarks_dict = {}
                
                # Duyệt qua từng bàn tay được phát hiện
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Xác định tay trái hay tay phải
                    handedness = results.multi_handedness[hand_idx].classification[0].label  # "Left" hoặc "Right"
                    
                    # Lưu landmarks vào dictionary
                    hand_landmarks_dict[handedness] = hand_landmarks
                    
                    # Vẽ các điểm mốc và các đường kết nối giữa chúng
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    
                    # Trích xuất đặc trưng tĩnh
                    static_features = extract_features(hand_landmarks)
                    
                    # Trích xuất đặc trưng động nếu có dữ liệu trước đó
                    dynamic_features = extract_dynamic_features(
                        hand_landmarks, previous_landmarks.get(handedness)
                    )
                    
                    # Lưu landmarks hiện tại cho frame tiếp theo
                    previous_landmarks[handedness] = hand_landmarks
                    
                    # Thu thập dữ liệu nếu đang trong chế độ thu thập
                    if collect_data and current_gesture_label:
                        recognizer.collect_data(static_features, current_gesture_label)
                        logger.info(f"Thu thập dữ liệu cho cử chỉ {current_gesture_label}")
                    
                    # Nếu đang ghi cử chỉ
                    if is_recording:
                        current_recording.append(static_features)
                    
                    # Sử dụng mô hình để dự đoán cử chỉ
                    prediction = recognizer.predict(static_features)
                    if prediction:
                        # Hiển thị dự đoán trên frame
                        cv2.putText(frame, f"Gesture: {prediction}", 
                                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    
                    # Phân cụm để phân loại chuyển động
                    cluster = recognizer.cluster(static_features)
                    if cluster is not None:
                        # Lưu vào lịch sử vị trí
                        position_history.append((static_features, cluster))
                        # Giới hạn kích thước lịch sử
                        if len(position_history) > max_history:
                            position_history.pop(0)
                        
                        # Vẽ màu tương ứng với cụm
                        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), 
                                 (255, 255, 0), (0, 255, 255), (255, 0, 255)]
                        color = colors[cluster % len(colors)]
                        
                        # Vẽ vòng tròn ở vị trí đầu ngón tay giữa
                        middle_tip = hand_landmarks.landmark[FINGER_TIPS["middle"]]
                        cv2.circle(frame, (int(middle_tip.x * frame.shape[1]), 
                                         int(middle_tip.y * frame.shape[0])), 
                                  15, color, -1)
                    
                    # Kiểm tra trạng thái các ngón tay
                    for finger in ["index", "middle", "ring"]:
                        # Tạo tên ngón tay (ví dụ: "left_index")
                        finger_name = f"{handedness.lower()}_{finger}"
                        
                        # Kiểm tra xem ngón tay có được hạ xuống không
                        if is_finger_down(hand_landmarks.landmark, 
                                         FINGER_TIPS[finger], 
                                         FINGER_MCP[finger]):
                            # Nếu ngón tay chưa được đánh dấu là hạ xuống
                            if not finger_state[finger_name]:
                                # Phát âm thanh với điều kiện cooldown
                                current_time = time.time()
                                if current_time - last_played[finger_name] > cooldown:
                                    sound_name = finger_to_sound.get(finger_name)
                                    if sound_name in sounds:
                                        sounds[sound_name].play()
                                        last_played[finger_name] = current_time
                                        logger.info(f"Phát âm thanh {sound_name} cho ngón {finger_name}")
                                
                                # Đánh dấu ngón tay đã được hạ xuống
                                finger_state[finger_name] = True
                        else:
                            # Đặt lại trạng thái nếu ngón tay không được hạ xuống
                            finger_state[finger_name] = False
                
                # Hiển thị trạng thái ngón tay lên frame
                for i, (finger, state) in enumerate(finger_state.items()):
                    status = "Hạ xuống" if state else "Hướng lên"
                    cv2.putText(frame, f"{finger}: {status}", 
                               (10, 120 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Hiển thị frame
            cv2.imshow(window_name, frame)
            
            # Xử lý phím nhấn
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                logger.info("Nhấn ESC, thoát chương trình.")
                break
            elif key == ord('r') or key == ord('R'):  # Bắt đầu/dừng ghi cử chỉ
                is_recording = not is_recording
                if is_recording:
                    current_recording = []
                    logger.info("Bắt đầu ghi cử chỉ...")
                else:
                    if current_recording:
                        gesture_name = f"gesture_{len(gesture_templates) + 1}"
                        gesture_templates[gesture_name] = current_recording
                        logger.info(f"Đã lưu cử chỉ {gesture_name}")
            elif key == ord('t') or key == ord('T'):  # Huấn luyện mô hình
                if recognizer.train() and recognizer.train_kmeans():
                    logger.info("Đã huấn luyện mô hình thành công")
                else:
                    logger.warning("Huấn luyện mô hình thất bại")
            elif key == ord('p') or key == ord('P'):  # Bật/tắt chế độ dự đoán
                collect_data = not collect_data
                if collect_data:
                    # Hỏi người dùng nhãn của cử chỉ
                    current_gesture_label = f"gesture_{len(recognizer.y) % 6 + 1}"
                    logger.info(f"Bắt đầu thu thập dữ liệu cho cử chỉ {current_gesture_label}")
                else:
                    current_gesture_label = None
                    logger.info("Dừng thu thập dữ liệu")
            
        # Giải phóng tài nguyên
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Đã giải phóng tài nguyên")

if __name__ == "__main__":
    main() 