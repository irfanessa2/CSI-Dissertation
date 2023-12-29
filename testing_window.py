import cv2
import mediapipe as mp
import tempfile
import os
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from roboflow import Roboflow
import numpy as np
from threading import Thread

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_INDICES = [133, 153, 154, 155, 145, 144, 163, 7, 33, 246, 161, 160, 159, 158, 157, 173]


def calculate_ear(eye):
    Minx,miny = np.min(eye,axis=0)
    Maxx,maxy = np.max(eye,axis=0)
    ear=(Maxx-Minx)/(maxy-miny)
    return 1/ear

def get_eye_landmarks(landmarks, indices):
    return [np.array([landmarks[i].x, landmarks[i].y]) for i in indices]

class CameraStream(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    blink_count_signal = pyqtSignal(int, float)
    ear_signal = pyqtSignal(float)
    blink_signal = pyqtSignal(int)

    def __init__(self, src=0, parent=None):
        super().__init__(parent)
        self.src = src
        self.stopped = False
        self.hands = mp_hands.Hands()
        self.face_mesh = mp_face_mesh.FaceMesh()
        self.blink_count = 0
        self.ear_threshold = 0.4  #0.21 before
        self.frame_skip = 1  # Skip every 5 frames
        self.resize_factor = 1  # Reduce frame size by 50%
        self.predictions = {}
        self.blinked = False
        self.prediction_thread = Thread()

        rf = Roboflow(api_key="WfcwDS8JerpXetYTTN9c")
        project = rf.workspace().project("temp-g8ijr")
        self.model = project.version(1).model
        
    def increment_blinks(self):
        self.blink_count+=1
        self.blink_signal.emit(self.blink_count)
        print(self.blink_count)

    
    def process_frame(self, frame):
        frame = cv2.resize(frame, None, fx=self.resize_factor, fy=self.resize_factor)

        # Roboflow prediction
        temp_file = 'tmp.jpg'
        cv2.imwrite(temp_file, frame)
        self.predictions = self.model.predict(temp_file, confidence=40, overlap=30).json()
        os.unlink(temp_file)
        print(f'predicted! {self.predictions.items()}')

    def draw_predictions(self, frame):

        if 'predictions' in self.predictions:
            for prediction in self.predictions['predictions']:
                x, y, w, h = prediction["x"], prediction["y"], prediction["width"], prediction["height"]
                cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)

        # Face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Hand detection
        frame_rgb = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Blink detection
        results = self.face_mesh.process(frame_rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_eye = get_eye_landmarks(face_landmarks.landmark, LEFT_EYE_INDICES)
                right_eye = get_eye_landmarks(face_landmarks.landmark, RIGHT_EYE_INDICES)

                for point in left_eye + right_eye:
                    cv2.circle(frame, (int(point[0] * frame.shape[1]), int(point[1] * frame.shape[0])), 1, (0, 255, 0), -1)

                left_ear = calculate_ear(left_eye)
                right_ear = calculate_ear(right_eye)
                ear = (left_ear + right_ear) / 2.0
                self.ear_signal.emit(ear)
                
                if ear < self.ear_threshold:
                    if not self.blinked:
                        self.blinked = True
                        self.increment_blinks()
                else:
                    self.blinked = False                 
        return frame

    def run(self):
        cap = cv2.VideoCapture(self.src)
        frame_count = 0
        INTERLEAVED = 10

        while not self.stopped:
            ret, frame = cap.read()
            if ret:
                # if not self.prediction_thread.is_alive():
                #     self.predictions_thread = Thread(target=self.process_frame(frame.copy()))
                #     self.predictions_thread.start()
                # frame_count += 1
                # if frame_count > INTERLEAVED:
                #     self.process_frame(frame)
                #     frame_count = 0
                frame = self.draw_predictions(frame)
                qimg = self.convert_to_qimage(frame)
                self.change_pixmap_signal.emit(qimg)
                
                

        cap.release()
        self.hands.close()
        self.face_mesh.close()

    def convert_to_qimage(self, frame):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        return QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888).scaled(640, 480, Qt.KeepAspectRatio)

    def stop(self):
        self.stopped = True
        self.wait()

class TestingWindow(QWidget):
    update = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Testing Window")
        self.resize(640, 480)
        self.layout = QVBoxLayout(self)
        self.image_label = QLabel(self)
        self.blink_count_label = QLabel("Blinks: 0, EAR: 0.0", self)
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.blink_count_label)
        self.setLayout(self.layout)
        self.blinks = 0
        self.ear = 0.0

        self.stream = CameraStream(0, self)
        self.stream.change_pixmap_signal.connect(self.update_image)
        #self.stream.blink_count_signal.connect(self.update_blink_count)
        self.stream.ear_signal.connect(self.update_ear)
        self.stream.blink_signal.connect(self.update_blink_count)
        self.update.connect(self.update_values)
        self.stream.start()

    @pyqtSlot(QImage)
    def update_image(self, image):
        self.image_label.setPixmap(QPixmap.fromImage(image))
    
    @pyqtSlot()
    def update_values(self):
        self.blink_count_label.setText(f"Blinks: {self.blinks}, EAR: {self.ear:.2f}")

    @pyqtSlot(int)
    def update_blink_count(self, count):
        self.blinks = count
        self.update.emit()
    
    
    @pyqtSlot(float)
    def update_ear(self, val):
        self.ear = val
        self.update.emit()

    def closeEvent(self, event):
        self.stream.stop()

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    mainWin = TestingWindow()
    mainWin.show()
    sys.exit(app.exec_())
