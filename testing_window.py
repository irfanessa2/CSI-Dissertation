from collections import deque
import time
import cv2
import mediapipe as mp
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QVBoxLayout,
    QWidget,
    QRadioButton,
    QHBoxLayout,
    QProgressBar,
)
import numpy as np
from threading import Thread

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE_INDICES = [
    362,
    382,
    381,
    380,
    374,
    373,
    390,
    249,
    263,
    466,
    388,
    387,
    386,
    385,
    384,
    398,
]
RIGHT_EYE_INDICES = [
    133,
    153,
    154,
    155,
    145,
    144,
    163,
    7,
    33,
    246,
    161,
    160,
    159,
    158,
    157,
    173,
]


class CameraStream(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    ear_signal = pyqtSignal(float)
    eyes_closed_signal = pyqtSignal()
    eyes_open_signal = pyqtSignal()
    latency = pyqtSignal(object)

    def __init__(self, src=0, parent=None):
        super().__init__(parent)
        self.src = src
        self.stopped = False
        self.hands = mp_hands.Hands()
        self.face_mesh = mp_face_mesh.FaceMesh()
        self.ear_threshold = 0.30  # 0.21 before
        self.predictions = {}
        self.blinked = False
        self.opened = False
        self.face_detection_thread = Thread()
        self.hand_detection_thread = Thread()
        self.blink_detection_thread = Thread()
        self.face_results = None
        self.hand_results = None
        self.blink_results = None
        self.do_face = False
        self.do_hands = False
        self.do_blink = False

    @pyqtSlot(bool)
    def set_do_hands(self, val):
        self.do_hands = val
        if not val and self.hand_detection_thread.is_alive():
            self.hand_detection_thread.join()  # blocking in MainThread, ensure the thread is very short

    @pyqtSlot(bool)
    def set_do_face(self, val):
        self.do_face = val
        if not val and self.face_detection_thread.is_alive():
            self.face_detection_thread.join()  # blocking in MainThread, ensure the thread is very short

    @pyqtSlot(bool)
    def set_do_blink(self, val):
        self.do_blink = val
        if not val and self.blink_detection_thread.is_alive():
            self.blink_detection_thread.join()  # blocking in MainThread, ensure the thread is very short

    def face_detection_worker(self, frame):
        # Face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.face_results = face_cascade.detectMultiScale(gray, 1.1, 4)

    def hand_detection_worker(self, frame):
        frame_rgb = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
        self.hand_results = self.hands.process(frame_rgb)

    def blink_detection_worker(self, frame):
        # Blink detection
        frame_rgb = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
        self.blink_results = self.face_mesh.process(frame_rgb)

    def draw_predictions(self, frame):
        if self.face_results is not None:
            for x, y, w, h in self.face_results:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Hand detection
        if self.hand_results is not None and self.hand_results.multi_hand_landmarks:
            for hand_landmarks in self.hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        if self.blink_results is not None and self.blink_results.multi_face_landmarks:
            for face_landmarks in self.blink_results.multi_face_landmarks:
                left_eye = self.get_eye_landmarks(
                    face_landmarks.landmark, LEFT_EYE_INDICES
                )
                right_eye = self.get_eye_landmarks(
                    face_landmarks.landmark, RIGHT_EYE_INDICES
                )

                for point in left_eye + right_eye:
                    cv2.circle(
                        frame,
                        (
                            int(point[0] * frame.shape[1]),
                            int(point[1] * frame.shape[0]),
                        ),
                        1,
                        (0, 255, 0),
                        -1,
                    )

                left_ear = self.calculate_ear(left_eye)
                right_ear = self.calculate_ear(right_eye)
                ear = (left_ear + right_ear) / 2.0
                self.ear_signal.emit(ear)

                if ear < self.ear_threshold:
                    if not self.blinked:
                        self.blinked = True
                        self.opened = False
                        self.eyes_closed_signal.emit()
                elif not self.opened:
                        self.opened = True
                        self.blinked = False
                        self.eyes_open_signal.emit()
        return frame

    def run(self):
        cap = cv2.VideoCapture(self.src)

        while not self.stopped:
            self.latency.emit(time.time_ns())
            ret, frame = cap.read()
            if ret:
                if self.do_hands:
                    if not self.hand_detection_thread.is_alive():
                        self.hand_detection_thread = Thread(
                            target=self.hand_detection_worker, args=(frame.copy(),)
                        )
                        self.hand_detection_thread.start()
                else:
                    self.hand_results = None

                if self.do_face:
                    if not self.face_detection_thread.is_alive():
                        self.face_detection_thread = Thread(
                            target=self.face_detection_worker, args=(frame.copy(),)
                        )
                        self.face_detection_thread.start()
                else:
                    self.face_results = None

                if self.do_blink:
                    if not self.blink_detection_thread.is_alive():
                        self.blink_detection_thread = Thread(
                            target=self.blink_detection_worker, args=(frame.copy(),)
                        )
                        self.blink_detection_thread.start()
                else:
                    self.blink_results = None

                frame = self.draw_predictions(frame)
                qimg = self.convert_to_qimage(frame)
                self.change_pixmap_signal.emit(qimg)

        cap.release()
        self.hands.close()
        self.face_mesh.close()

    def stop(self):
        self.stopped = True
        self.wait()

    @staticmethod
    def calculate_ear(eye):
        min_x, min_y = np.min(eye, axis=0)
        max_x, max_y = np.max(eye, axis=0)
        return (max_y - min_y) / (max_x - min_x)

    @staticmethod
    def get_eye_landmarks(landmarks, indices):
        return [np.array([landmarks[i].x, landmarks[i].y]) for i in indices]
        # return [np.array([lm.x,lm.y]) for lm in landmarks] # uncomment for full face points

    @staticmethod
    def convert_to_qimage(frame):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        return QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888).scaled(
            640, 480, Qt.KeepAspectRatio
        )


class TestingWindow(QWidget):
    update = pyqtSignal()
    do_face = pyqtSignal(bool)
    do_hands = pyqtSignal(bool)
    do_blink = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Testing Window")
        self.resize(900, 480)
        self.blinks = 0
        self.ear = 0.0
        self._last_frame_ts = 0
        self.fps = deque(maxlen=10)
        self.latency = deque(maxlen=10)
        self.eyes_closed_timer = QTimer(self)
        self.eyes_closed_timer.timeout.connect(self.sleep_bar_inc_method)
        self.eyes_closed_timer.setInterval(1)
        self.eyes_opened_timer = QTimer(self)
        self.eyes_opened_timer.timeout.connect(self.sleep_bar_dec_method)
        self.eyes_opened_timer.setInterval(1)
        self.image_label = QLabel(self)
        self.blink_count_label = QLabel(self)
        self.stream = CameraStream(0, self)
        self.fps_label = QLabel('FPS: 0',self)
        self.latency_label = QLabel('Latency: 0',self)
        self.asleep_label = QLabel('ASLEEP',self)
        self.asleep_label.setStyleSheet('color:red')
        self.asleep_label.hide()

        main_layout = QHBoxLayout()
        stats_layout = QVBoxLayout()
        frame_stats_layout = QHBoxLayout()

        self.face_toggle = QRadioButton("Face Detection", self, autoExclusive=False)
        self.hands_toggle = QRadioButton("Hands Detection", self, autoExclusive=False)
        self.blink_toggle = QRadioButton("Blink Detection", self, autoExclusive=False)
        
        self.sleep_bar = QProgressBar(self)
        self.sleep_bar.setMaximum(3*1000)
        self.sleep_bar.setValue(0)

        frame_stats_layout.addWidget(self.fps_label)
        frame_stats_layout.addWidget(self.latency_label)

        stats_layout.addWidget(self.face_toggle)
        stats_layout.addWidget(self.hands_toggle)
        stats_layout.addWidget(self.blink_toggle)
        stats_layout.addWidget(QWidget())
        stats_layout.addWidget(self.sleep_bar)
        stats_layout.addWidget(self.asleep_label)
        stats_layout.addWidget(self.blink_count_label)
        stats_layout.addLayout(frame_stats_layout)
        stats_layout.setAlignment(Qt.AlignTop)

        main_layout.addWidget(self.image_label)
        main_layout.addLayout(stats_layout)

        self.setLayout(main_layout)

        self.face_toggle.toggled.connect(
            lambda: self.do_face.emit(self.face_toggle.isChecked())
        )
        self.hands_toggle.toggled.connect(
            lambda: self.do_hands.emit(self.hands_toggle.isChecked())
        )
        self.blink_toggle.toggled.connect(
            lambda: self.do_blink.emit(self.blink_toggle.isChecked())
        )

        self.do_face.connect(self.stream.set_do_face)
        self.do_hands.connect(self.stream.set_do_hands)
        self.do_blink.connect(self.stream.set_do_blink)
        
        self.stream.latency.connect(self.latency_update)
        self.stream.change_pixmap_signal.connect(self.update_image)
        self.stream.ear_signal.connect(self.update_ear)
        self.stream.eyes_closed_signal.connect(self.update_blink_count)
        self.stream.eyes_closed_signal.connect(self.start_closed_eyes_timer)
        self.stream.eyes_open_signal.connect(self.start_open_eyes_timer)
        self.update.connect(self.update_values)
        self.stream.start()
        self.update_values()
        
    def start_closed_eyes_timer(self):
        self.stop_all_timers()
        self.eyes_closed_timer.start()

    def start_open_eyes_timer(self):
        self.stop_all_timers()
        self.eyes_opened_timer.start()
        
    def stop_all_timers(self):
        if self.eyes_closed_timer.isActive():
            self.eyes_closed_timer.stop()
        if self.eyes_opened_timer.isActive():
            self.eyes_opened_timer.stop()
        
    def sleep_bar_inc_method(self):
        if self.eyes_opened_timer.isActive():
            self.eyes_opened_timer.stop()
        if self.sleep_bar.value() == self.sleep_bar.maximum():
            self.asleep_label.show()
            return
        self.sleep_bar.setValue(self.sleep_bar.value() + 1)
    
    def sleep_bar_dec_method(self):
        self.asleep_label.hide()
        if self.eyes_closed_timer.isActive():
            self.eyes_closed_timer.stop()
        if self.sleep_bar.value() == self.sleep_bar.minimum():
            return
        self.sleep_bar.setValue(self.sleep_bar.value() - 1)

        
    @pyqtSlot(object)
    def latency_update(self, ts):
        self.frame_ts = ts

    @pyqtSlot(QImage)
    def update_image(self, image):
        frame_ts = time.time_ns()
        self.fps.append(frame_ts-self._last_frame_ts)
        self._last_frame_ts = frame_ts
        self.latency.append(time.time_ns() - self.frame_ts)
        self.latency_label.setText(f"Latency: {np.mean(self.latency)/1e6:.2f}ms")
        self.fps_label.setText(f"FPS: {1e9/np.mean(self.fps):.2f}")
        self.image_label.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot()
    def update_values(self):
        self.blink_count_label.setText(f"Blinks: {self.blinks}, EAR: {self.ear:.2f}")

    @pyqtSlot()
    def update_blink_count(self):
        self.blinks += 1
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
