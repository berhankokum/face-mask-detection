import sys
import cv2
import os
import pickle
import numpy as np
from keras_facenet import FaceNet
from tensorflow.keras.models import load_model
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton,
                             QVBoxLayout, QLineEdit, QMessageBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

USERS_DIR = "users"
os.makedirs(USERS_DIR, exist_ok=True)

# OpenCV DNN face detector
prototxt_path = os.path.join("face_detector", "deploy.prototxt")
weights_path = os.path.join("face_detector", "res10_300x300_ssd_iter_140000.caffemodel")
face_net = cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)

class MaskDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Maske Tespiti ve Kullanıcı Girişi (FaceNet)")
        self.setGeometry(100, 100, 800, 600)

        self.video_label = QLabel(self)
        self.video_label.setFixedSize(640, 480)

        self.name_input = QLineEdit(self)
        self.name_input.setAlignment(Qt.AlignCenter)
        self.name_input.setPlaceholderText("Kullanıcı adı girin")

        self.register_button = QPushButton("Kayıt Ol")
        self.login_button = QPushButton("Giriş Yap")
        self.start_register_button = QPushButton("Kaydı Başlat")
        self.logout_button = QPushButton("Çıkış Yap")
        self.back_to_menu_button = QPushButton("Ana Menüye Dön")

        self.register_button.clicked.connect(self.start_camera_and_register)
        self.login_button.clicked.connect(self.start_camera_and_login)
        self.logout_button.clicked.connect(self.logout_user)
        self.start_register_button.clicked.connect(self.register_user)
        self.back_to_menu_button.clicked.connect(self.return_to_main_menu)

        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.register_button)
        self.layout.addWidget(self.login_button)
        self.layout.addWidget(self.name_input)
        self.layout.addWidget(self.start_register_button)
        self.layout.addWidget(self.logout_button)
        self.layout.addWidget(self.back_to_menu_button)
        self.layout.addWidget(self.video_label)

        self.setLayout(self.layout)

        self.capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.current_frame = None
        self.logged_in = False
        self.logged_in_user = None

        self.mask_model = load_model("mask_detector_model.h5")
        self.facenet = FaceNet()
        self.known_users = []
        self.last_check_time = 0
        self.mask_status_cache = {}

        self.reset_to_main_menu()

    def reset_to_main_menu(self):
        self.clear_interface()
        self.register_button.show()
        self.login_button.setText("Giriş Yap")
        self.login_button.clicked.disconnect()
        self.login_button.clicked.connect(self.start_camera_and_login)
        self.login_button.show()

    def clear_interface(self):
        self.name_input.hide()
        self.name_input.clear()
        self.login_button.hide()
        self.logout_button.hide()
        self.start_register_button.hide()
        self.back_to_menu_button.hide()
        self.register_button.hide()

    def return_to_main_menu(self):
        self.release_camera()
        self.reset_to_main_menu()

    def init_camera(self):
        if self.capture is None:
            self.capture = cv2.VideoCapture(0)
            self.timer.start(1000)

    def release_camera(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None
            self.timer.stop()
            self.video_label.clear()

    def start_camera_and_login(self):
        self.init_camera()
        self.clear_interface()
        self.name_input.show()
        self.name_input.setFocus()
        self.login_button.setText("Giriş Yap")
        self.login_button.show()
        self.login_button.clicked.disconnect()
        self.login_button.clicked.connect(self.login_user)

    def start_camera_and_register(self):
        self.init_camera()
        self.clear_interface()
        self.name_input.show()
        self.name_input.setFocus()
        self.start_register_button.show()

    def detect_faces_dnn(self, frame):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()
        boxes = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                boxes.append(box.astype("int"))
        return boxes

    def detect_single_face(self, rgb_frame):
        boxes = self.detect_faces_dnn(rgb_frame)
        if len(boxes) != 1:
            return None
        (startX, startY, endX, endY) = boxes[0]
        face = rgb_frame[startY:endY, startX:endX]
        face = cv2.resize(face, (160, 160))
        return face

    def register_user(self):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Uyarı", "Lütfen bir kullanıcı adı girin.")
            return

        if self.current_frame is not None:
            QMessageBox.information(self, "Bilgi", "1. Aşama: Maskesiz yüzünüzle kayıt olun.")
            rgb_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            boxes = self.detect_faces_dnn(rgb_frame)
            if len(boxes) != 1:
                QMessageBox.warning(self, "Uyarı", "Lütfen yalnızca bir yüzle kaydolun.")
                return
            (startX, startY, endX, endY) = boxes[0]
            face = rgb_frame[startY:endY, startX:endX]
            face = cv2.resize(face, (160, 160))
            emb_nomask = self.facenet.embeddings([face])[0]

            QMessageBox.information(self, "Bilgi", "2. Aşama: Maskeli halinizle tekrar bakın ve 'Tamam' deyin.")
            QMessageBox.information(self, "Hazır Mısınız?", "Maskenizi taktıktan sonra OK tuşuna basın.")

            # Yeni bir kare al
            ret, frame = self.capture.read()
            if not ret:
                QMessageBox.warning(self, "Hata", "Kameradan görüntü alınamadı.")
                return
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = self.detect_faces_dnn(rgb_frame)
            if len(boxes) != 1:
                QMessageBox.warning(self, "Uyarı", "Maskeli halde yalnızca bir yüz olmalı.")
                return
            (startX, startY, endX, endY) = boxes[0]
            face = rgb_frame[startY:endY, startX:endX]
            face = cv2.resize(face, (160, 160))
            emb_mask = self.facenet.embeddings([face])[0]

            user_data = {"name": name, "embeddings": [emb_nomask, emb_mask]}
            with open(os.path.join(USERS_DIR, f"{name}.pkl"), "wb") as f:
                pickle.dump(user_data, f)

            QMessageBox.information(self, "Başarılı", f"{name} için veriler kaydedildi.")


    def detect_faces_dnn(self, frame):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()
        boxes = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                boxes.append(box.astype("int"))
        return boxes


    def login_user(self):
        self.known_users.clear()
        for filename in os.listdir(USERS_DIR):
            if filename.endswith(".pkl"):
                with open(os.path.join(USERS_DIR, filename), "rb") as f:
                    self.known_users.append(pickle.load(f))

        if not self.known_users:
            QMessageBox.warning(self, "Hata", "Hiç kayıtlı kullanıcı yok.")
            return

        if self.current_frame is not None:
            rgb_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            face = self.detect_single_face(rgb_frame)
            if face is None:
                QMessageBox.warning(self, "Hata", "Yüz algılanamadı.")
                return
            input_emb = self.facenet.embeddings([face])[0]

            best_match = None
            best_distance = 1.0
            for user in self.known_users:
                for emb in user["embeddings"]:
                    dist = np.linalg.norm(input_emb - emb)
                    if dist < best_distance:
                        best_distance = dist
                        best_match = user["name"]

            if best_distance < 0.8:
                QMessageBox.information(self, "Başarılı", f"{best_match} olarak giriş yapıldı.")
                self.clear_interface()
                self.back_to_menu_button.show()
            else:
                QMessageBox.warning(self, "Hata", "Yüz eşleşmedi.")

    def update_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            return

        self.current_frame = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_time = cv2.getTickCount() / cv2.getTickFrequency()

        run_check = current_time - self.last_check_time >= 1.0
        if run_check:
            self.last_check_time = current_time
            boxes = self.detect_faces_dnn(rgb_frame)
            self.last_boxes = boxes
        else:
            boxes = getattr(self, 'last_boxes', [])

        for (startX, startY, endX, endY) in boxes:
            face = rgb_frame[startY:endY, startX:endX]
            if face.size == 0:
                continue
            try:
                face_resized = cv2.resize(face, (160, 160))
            except:
                continue
            embedding = self.facenet.embeddings([face_resized])[0]

            label = "Bilinmiyor"
            color = (255, 255, 0)

            min_dist = 1.0
            best_match = None
            for user in self.known_users:
                for emb in user['embeddings']:
                    dist = np.linalg.norm(embedding - emb)
                    if dist < min_dist:
                        min_dist = dist
                        best_match = user['name']

            if best_match and min_dist < 0.8:
                self.logged_in = True
                self.logged_in_user = best_match

                face_id = (startX, startY, endX, endY)
                if run_check:
                    try:
                        face_crop = frame[startY:endY, startX:endX]
                        resized = cv2.resize(face_crop, (224, 224))
                        face_array = resized.astype("float32") / 255.0
                        face_array = np.expand_dims(face_array, axis=0)
                        (mask, withoutMask) = self.mask_model.predict(face_array, verbose=0)[0]
                        mask_status = "Maskeli" if mask > withoutMask else "Maskesiz"
                        self.mask_status_cache[face_id] = (mask_status, (0, 255, 0) if mask > withoutMask else (0, 0, 255))
                    except:
                        continue

                if face_id in self.mask_status_cache:
                    mask_status, color = self.mask_status_cache[face_id]
                    label = f"{best_match}: {mask_status}"

            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def logout_user(self):
        self.logged_in = False
        self.logged_in_user = None
        self.return_to_main_menu()

    def closeEvent(self, event):
        self.release_camera()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MaskDetectionApp()
    window.show()
    sys.exit(app.exec_())
