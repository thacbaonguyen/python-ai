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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

try:
    pygame.mixer.init()
    logger.info("init pygame mixer")
except Exception as e:
    logger.error(f"error init pygame mixer: {e}")
    exit(1)

MODEL_DIR = 'models'
DATA_DIR = 'data'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

sounds = {}
sound_files = {
    "mi": "mi.wav",  # Ngón trỏ tay trái
    "re": "re.wav",  # Ngón giữa tay trái
    "do": "do.wav",  # Ngón áp út tay trái
    "fa": "fa.wav",  # Ngón trỏ tay phải
    "sol": "sol.wav",  # Ngón giữa tay phải
    "la": "la.wav"  # Ngón áp út tay phải
}

try:
    for name, file in sound_files.items():
        file_path = os.path.join(DATA_DIR, file)
        if os.path.exists(file_path):
            sounds[name] = pygame.mixer.Sound(file_path)
        else:
            logger.warning(f"file {file_path} not found")
    logger.info("load sound files")
except Exception as e:
    logger.error(f"error load sound files: {e}")

finger_to_sound = {
    "left_index": "mi",
    "left_middle": "re",
    "left_ring": "do",
    "right_index": "fa",
    "right_middle": "sol",
    "right_ring": "la"
}

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

NUM_FEATURES = 63  # 21 landmarks x 3 dimensions (x,y,z)

def extract_features(hand_landmarks):
    features = []
    for landmark in hand_landmarks.landmark:
        features.extend([landmark.x, landmark.y, landmark.z])
    return features


def extract_dynamic_features(current_landmarks, previous_landmarks):
    features = []
    if previous_landmarks:
        for i, landmark in enumerate(current_landmarks.landmark):
            prev_landmark = previous_landmarks.landmark[i]
            velocity_x = landmark.x - prev_landmark.x
            velocity_y = landmark.y - prev_landmark.y
            velocity_z = landmark.z - prev_landmark.z
            features.extend([velocity_x, velocity_y, velocity_z])
    else:
        features = [0] * (21 * 3)
    return features


def is_finger_down(landmarks, finger_tip, finger_mcp):
    return landmarks[finger_tip].y > landmarks[finger_mcp].y

class HandGestureRecognizer:
    def __init__(self):

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

        self.kmeans = KMeans(n_clusters=6, random_state=42)

        self.scaler = StandardScaler()

        self.X = []
        self.y = []

        self.cluster_data = []

        self.model_path = os.path.join(MODEL_DIR, "hand_gesture_model.pkl")
        self.kmeans_path = os.path.join(MODEL_DIR, "kmeans_model.pkl")
        self.scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

        self.load_models()

    def collect_data(self, features, label):
        """collect data"""
        self.X.append(features)
        self.y.append(label)
        self.cluster_data.append(features)

    def train(self):
        """train model"""
        if len(self.X) < 10 or len(set(self.y)) < 2:
            logger.warning("not enough data to train model")
            return False

        try:

            X_scaled = self.scaler.fit_transform(self.X)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, self.y, test_size=0.2, random_state=42)

            self.model.fit(X_train, y_train)

            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"model accuracy: {accuracy}")

            self.save_models()

            return True
        except Exception as e:
            logger.error(f"error train model: {e}")
            return False

    def train_kmeans(self):
        """train kmeans model"""
        if len(self.cluster_data) < 10:
            logger.warning("not enough data to train kmeans model")
            return False

        try:

            data_scaled = self.scaler.transform(self.cluster_data)

            self.kmeans.fit(data_scaled)

            self.save_models()

            return True
        except Exception as e:
            logger.error(f"error train kmeans model: {e}")
            return False

    def predict(self, features):
        """predict gesture using model"""
        try:

            features_scaled = self.scaler.transform([features])

            prediction = self.model.predict(features_scaled)
            return prediction[0]
        except Exception as e:
            logger.error(f"error predict gesture: {e}")
            return None

    def cluster(self, features):
        """cluster gesture using kmeans model"""
        try:

            features_scaled = self.scaler.transform([features])

            cluster = self.kmeans.predict(features_scaled)
            return cluster[0]
        except Exception as e:
            logger.error(f"error cluster gesture: {e}")
            return None

    def save_models(self):
        """save model to file"""
        try:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.kmeans, self.kmeans_path)
            joblib.dump(self.scaler, self.scaler_path)
            logger.info("save models")
        except Exception as e:
            logger.error(f"error save models: {e}")

    def load_models(self):
        """load model from file"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logger.info("load model")

            if os.path.exists(self.kmeans_path):
                self.kmeans = joblib.load(self.kmeans_path)
                logger.info("load kmeans model")

            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                logger.info("load scaler")

            return True
        except Exception as e:
            logger.error(f"error load models: {e}")
            return False


def main():
    recognizer = HandGestureRecognizer()
    logger.info(f"use camera with ID: {video_source}")

    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        logger.error(f"can't open camera with ID: {video_source}")
        alternative_source = 1
        logger.info(f"try camera with ID: {alternative_source}")
        cap = cv2.VideoCapture(alternative_source)
        if not cap.isOpened():
            logger.error(f"can't open any camera")
            return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"camera resolution: {width}x{height}")

    finger_state = {
        "left_index": False,
        "left_middle": False,
        "left_ring": False,
        "right_index": False,
        "right_middle": False,
        "right_ring": False
    }

    last_played = {finger: 0 for finger in finger_state.keys()}
    cooldown = 0.5  # time between plays (seconds)

    previous_landmarks = {
        "Left": None,
        "Right": None
    }

    collect_data = False
    current_gesture_label = None

    position_history = []
    max_history = 100

    gesture_templates = {}
    current_recording = []
    is_recording = False

    window_name = "Choi nhac bang cu chi ngon tay"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7) as hands:

        logger.info("start processing video from camera...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.error("can't read frame from camera.")
                break

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb_frame)


            cv2.putText(frame, "Press R: Record gesture | T: Train model | P: Prediction mode",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if is_recording:
                cv2.putText(frame, "Recording Gesture...", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if results.multi_hand_landmarks:
                hand_landmarks_dict = {}
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    handedness = results.multi_handedness[hand_idx].classification[0].label  # "Left" hoặc "Right"

                    hand_landmarks_dict[handedness] = hand_landmarks
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                    static_features = extract_features(hand_landmarks)
                    dynamic_features = extract_dynamic_features(
                        hand_landmarks, previous_landmarks.get(handedness)
                    )

                    previous_landmarks[handedness] = hand_landmarks

                    if collect_data and current_gesture_label:
                        recognizer.collect_data(static_features, current_gesture_label)
                        logger.info(f"collect data for gesture {current_gesture_label}")

                    if is_recording:
                        current_recording.append(static_features)
                    prediction = recognizer.predict(static_features)
                    if prediction:
                        cv2.putText(frame, f"Gesture: {prediction}",
                                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cluster = recognizer.cluster(static_features)
                    if cluster is not None:
                        position_history.append((static_features, cluster))
                        if len(position_history) > max_history:
                            position_history.pop(0)

                        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
                                  (255, 255, 0), (0, 255, 255), (255, 0, 255)]
                        color = colors[cluster % len(colors)]

                        middle_tip = hand_landmarks.landmark[FINGER_TIPS["middle"]]
                        cv2.circle(frame, (int(middle_tip.x * frame.shape[1]),
                                           int(middle_tip.y * frame.shape[0])),
                                   15, color, -1)

                    for finger in ["index", "middle", "ring"]:
                        finger_name = f"{handedness.lower()}_{finger}"

                        if is_finger_down(hand_landmarks.landmark,
                                          FINGER_TIPS[finger],
                                          FINGER_MCP[finger]):
                            if not finger_state[finger_name]:
                                current_time = time.time()
                                if current_time - last_played[finger_name] > cooldown:
                                    sound_name = finger_to_sound.get(finger_name)
                                    if sound_name in sounds:
                                        sounds[sound_name].play()
                                        last_played[finger_name] = current_time
                                        logger.info(f"play sound {sound_name} for finger {finger_name}")

                                finger_state[finger_name] = True
                        else:
                            finger_state[finger_name] = False

                for i, (finger, state) in enumerate(finger_state.items()):
                    status = "Ha xuong" if state else "Huong len"
                    cv2.putText(frame, f"{finger}: {status}",
                                (10, 120 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1)
            if key == 27:  
                logger.info("press ESC, exit program.")
                break
            elif key == ord('r') or key == ord('R'): 
                is_recording = not is_recording
                if is_recording:
                    current_recording = []
                    logger.info("start recording gesture...")
                else:
                    if current_recording:
                        gesture_name = f"gesture_{len(gesture_templates) + 1}"
                        gesture_templates[gesture_name] = current_recording
                        logger.info(f"save gesture {gesture_name}")
            elif key == ord('t') or key == ord('T'): 
                if recognizer.train() and recognizer.train_kmeans():
                    logger.info("train model successfully")
                else:
                    logger.warning("train model failed")
            elif key == ord('p') or key == ord('P'):  
                collect_data = not collect_data
                if collect_data:

                    current_gesture_label = f"gesture_{len(recognizer.y) % 6 + 1}"
                    logger.info(f"start collecting data for gesture {current_gesture_label}")
                else:
                    current_gesture_label = None
                    logger.info("stop collecting data")

        cap.release()
        cv2.destroyAllWindows()
        logger.info("release resources")


if __name__ == "__main__":
    main() 