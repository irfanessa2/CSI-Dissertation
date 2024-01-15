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
import pyqtgraph as pg
from datetime import datetime, timedelta
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


UPPER_LIP_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]


LOWER_LIP_INDICES = [
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    308,
    415,
    310,
    311,
    312,
    13,
    82,
]


class CameraStream(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    ear_signal = pyqtSignal(float)
    mar_signal = pyqtSignal(float)
    eyes_closed_signal = pyqtSignal()
    eyes_open_signal = pyqtSignal()
    latency_signal = pyqtSignal(object)

    def __init__(self, src=0, parent=None):
        super().__init__(parent)
        self.src = src
        self.stopped = False
        self.hands = mp_hands.Hands()
        self.face_mesh = mp_face_mesh.FaceMesh()
        self.ear_threshold = 0.40  # 0.21 before
        self.blinked = False
        self.opened = False
        self.face_feature_detection_thread = Thread()
        self.hand_detection_thread = Thread()
        self.face_results = None
        self.hand_results = None
        self.do_face = False
        self.do_hands = False
        self.do_blink = False
        self.do_yawn = False

    @pyqtSlot(bool)
    def set_do_hands(self, val):
        self.do_hands = val

    @pyqtSlot(bool)
    def set_do_face(self, val):
        self.do_face = val

    @pyqtSlot(bool)
    def set_do_yawn(self, val):
        self.do_yawn = val

    @pyqtSlot(bool)
    def set_do_blink(self, val):
        self.do_blink = val

    def hand_detection_worker(self, frame):
        frame_rgb = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
        self.hand_results = self.hands.process(frame_rgb)

    def face_feature_detection_worker(self, frame):
        # Blink detection
        frame_rgb = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
        self.face_results = self.face_mesh.process(frame_rgb)

    def draw_hands(self, frame):
        # Hand detection
        if self.hand_results is not None and self.hand_results.multi_hand_landmarks:
            for hand_landmarks in self.hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

    def draw_face_features(self, frame):
        # face detection
        if self.face_results is not None and self.face_results.multi_face_landmarks:
            for face_landmarks in self.face_results.multi_face_landmarks:
                if self.do_face:
                    for point in [
                        np.array([lm.x, lm.y]) for lm in face_landmarks.landmark
                    ]:
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

                feature_points = []
                if self.do_blink:
                    left_eye = self.get_feature_landmarks(
                        face_landmarks.landmark, LEFT_EYE_INDICES
                    )
                    right_eye = self.get_feature_landmarks(
                        face_landmarks.landmark, RIGHT_EYE_INDICES
                    )
                    left_ear = self.calculate_ar(left_eye)
                    right_ear = self.calculate_ar(right_eye)
                    ear = (left_ear + right_ear) / 2.0
                    self.ear_signal.emit(ear)

                    feature_points = feature_points + right_eye + left_eye

                    if ear < self.ear_threshold:
                        if not self.blinked:
                            self.blinked = True
                            self.opened = False
                            self.eyes_closed_signal.emit()
                    elif not self.opened:
                        self.opened = True
                        self.blinked = False
                        self.eyes_open_signal.emit()

                if self.do_yawn:
                    mouth = self.get_feature_landmarks(
                        face_landmarks.landmark, UPPER_LIP_INDICES + LOWER_LIP_INDICES
                    )
                    mar = self.calculate_ar(mouth)
                    self.mar_signal.emit(mar)

                    feature_points = feature_points + mouth

                for point in feature_points:
                    cv2.circle(
                        frame,
                        (
                            int(point[0] * frame.shape[1]),
                            int(point[1] * frame.shape[0]),
                        ),
                        1,
                        (0, 0, 255),
                        -1,
                    )

        return frame

    def run(self):
        cap = cv2.VideoCapture(self.src)

        while not self.stopped:
            self.latency_signal.emit(time.time_ns())
            ret, frame = cap.read()
            if ret:
                if self.do_hands:
                    self.service_thread(
                        "hand_detection_thread",
                        self.hand_detection_worker,
                        (frame.copy(),),
                    )
                    self.draw_hands(frame)
                else:
                    self.hand_results = None

                if self.do_face or self.do_blink or self.do_yawn:
                    self.service_thread(
                        "face_feature_detection_thread",
                        self.face_feature_detection_worker,
                        (frame.copy(),),
                    )
                    self.draw_face_features(frame)
                else:
                    self.face_results = None

                qimg = self.convert_to_qimage(frame)
                self.change_pixmap_signal.emit(qimg)

        cap.release()
        self.hands.close()
        self.face_mesh.close()

    def service_thread(self, thread, method, args=()):
        if not getattr(self, thread).is_alive():
            setattr(
                self,
                thread,
                Thread(target=method, args=args, name=f"{method.__name__}"),
            )
            getattr(self, thread).start()

    def stop(self):
        self.stopped = True
        self.wait()

    @staticmethod
    def calculate_ar(eye):
        min_x, min_y = np.min(eye, axis=0)
        max_x, max_y = np.max(eye, axis=0)
        return (max_y - min_y) / (max_x - min_x)

    @staticmethod
    def get_feature_landmarks(landmarks, indices):
        return [np.array([landmarks[i].x, landmarks[i].y]) for i in indices]

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
    do_yawn = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Testing Window")
        self.resize(900, 480)
        self.blinks = 0
        self.start_time = datetime.now()  # Initialize the start time
        self.ear = 0.0
        self.yawns = 0
        self._last_frame_ts = 0
        self.fps = deque(maxlen=10)
        self.latency = deque(maxlen=10)
        self.eyes_closed_timer = QTimer(self)
        self.eyes_closed_timer.timeout.connect(self.sleep_bar_inc_method)
        self.eyes_closed_timer.setInterval(1)
        self.eyes_opened_timer = QTimer(self)
        self.eyes_opened_timer.timeout.connect(self.sleep_bar_dec_method)
        self.eyes_opened_timer.setInterval(1)

        self.time_data = deque(
            maxlen=300
        )  # Store up to the last 300 data points for time
        self.bpm_data = deque(
            maxlen=300
        )  # Store up to the last 300 data points for BPM
        self.ypm_data = deque(
            maxlen=300
        )  # Store up to the last 300 data points for YPM

        # Initialize plot widget
        self.bpm_plot_widget = pg.PlotWidget(self)
        self.bpm_curve = self.bpm_plot_widget.plot(pen="y")
        self.ypm_plot_widget = pg.PlotWidget(self)
        self.ypm_curve = self.ypm_plot_widget.plot(pen="r")

        self.image_label = QLabel(self)
        self.blink_count_label = QLabel(self)
        self.yawn_count_label = QLabel(self)
        self.stream = CameraStream(0, self)  # Replace with your actual CameraStream
        self.fps_label = QLabel("FPS: 0", self)
        self.bpm_label = QLabel("BPM: 0", self)  # Use for displaying text BPM
        self.ypm_label = QLabel("YPM: 0", self)  # Use for displaying text YPM

        self.mar = QLabel("Mouth AR:", self)
        self.yawn_type = QLabel(self)
        self.latency_label = QLabel("Latency: 0", self)
        self.asleep_label = QLabel("ASLEEP", self)
        self.asleep_label.setStyleSheet("color:red")
        self.asleep_label.hide()

        main_layout = QHBoxLayout()
        stats_layout = QVBoxLayout()
        frame_stats_layout = QHBoxLayout()
        yawn_stats_layout = QHBoxLayout()

        self.face_toggle = QRadioButton("Face Detection", self, autoExclusive=False)
        self.hands_toggle = QRadioButton("Hands Detection", self, autoExclusive=False)
        self.blink_toggle = QRadioButton("Blink Detection", self, autoExclusive=False)
        self.yawn_toggle = QRadioButton("Yawn Detection", self, autoExclusive=False)

        self.is_yawning = False

        self.sleep_bar = QProgressBar(self)
        self.sleep_bar.setMaximum(3 * 1000)
        self.sleep_bar.setValue(0)

        yawn_stats_layout.addWidget(self.mar)
        yawn_stats_layout.addWidget(self.yawn_type)

        frame_stats_layout.addWidget(self.fps_label)
        frame_stats_layout.addWidget(self.latency_label)
        frame_stats_layout.addWidget(self.bpm_label)
        frame_stats_layout.addWidget(self.ypm_label)

        stats_layout.addWidget(self.face_toggle)
        stats_layout.addWidget(self.hands_toggle)
        stats_layout.addWidget(self.blink_toggle)
        stats_layout.addWidget(self.yawn_toggle)
        stats_layout.addWidget(QWidget())  # Placeholder widget
        stats_layout.addWidget(self.sleep_bar)
        stats_layout.addWidget(self.asleep_label)
        stats_layout.addWidget(self.blink_count_label)
        stats_layout.addLayout(yawn_stats_layout)
        stats_layout.addLayout(frame_stats_layout)
        stats_layout.addWidget(
            self.bpm_plot_widget
        )  # Add the plot widget to the stats layout
        stats_layout.addWidget(self.ypm_plot_widget)

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

        self.yawn_toggle.toggled.connect(
            lambda: self.do_yawn.emit(self.yawn_toggle.isChecked())
        )

        self.do_face.connect(self.stream.set_do_face)
        self.do_hands.connect(self.stream.set_do_hands)
        self.do_blink.connect(self.stream.set_do_blink)
        self.do_yawn.connect(self.stream.set_do_yawn)
        
        self.stream.latency_signal.connect(self.latency_update)
        self.stream.change_pixmap_signal.connect(self.update_image)
        self.stream.ear_signal.connect(self.update_ear)
        self.stream.eyes_closed_signal.connect(self.update_blink_count)
        self.stream.eyes_closed_signal.connect(self.start_closed_eyes_timer)
        self.stream.eyes_open_signal.connect(self.start_open_eyes_timer)
        self.stream.mar_signal.connect(self.update_mar)
        self.update.connect(self.update_values)
        self.stream.start()
        self.update_values()

    @pyqtSlot(float)
    def update_mar(self, val):
        self.mar.setText(f"Mouth AR: {val:.2f}")
        if val > 1.0:
            yawn_type = "YAWNING"
            if not self.is_yawning:  # Check if transition to yawning state
                self.is_yawning = True
                self.update_yawn_count()
        elif val < 0.6:
            yawn_type = "CLOSED"
            self.is_yawning = False  # Reset yawn state when mouth is closed
        else:
            yawn_type = "TALKING"
            self.is_yawning = False  # Reset yawn state when talking
        self.yawn_type.setText(yawn_type)

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
        self.fps.append(frame_ts - self._last_frame_ts)
        self._last_frame_ts = frame_ts
        self.latency.append(time.time_ns() - self.frame_ts)
        self.latency_label.setText(f"Latency: {np.mean(self.latency)/1e6:.2f}ms")
        self.fps_label.setText(f"FPS: {1e9/np.mean(self.fps):.2f}")
        self.update_bpm_label()
        self.update_ypm_label()
        self.image_label.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot()
    def update_values(self):
        self.blink_count_label.setText(f"Blinks: {self.blinks}, EAR: {self.ear:.2f}")

    @pyqtSlot()
    def update_blink_count(self):
        self.blinks += 1  # Increment the blink count
        self.update.emit()  # Emit signal to update GUI (if needed)

    def calculate_average_bpm(self):
        elapsed_time_minutes = (
            datetime.now() - self.start_time
        ).total_seconds() / 60  # Convert elapsed time to minutes
        if elapsed_time_minutes > 0:
            return self.blinks / elapsed_time_minutes  # Calculate average BPM
        else:
            return 0

    def update_bpm_label(self):
        # Call this method at a regular interval to update the plot
        average_bpm = self.calculate_average_bpm()
        current_time = (
            datetime.now() - self.start_time
        ).total_seconds() / 60  # Current time in minutes
        self.time_data.append(current_time)
        self.bpm_data.append(average_bpm)
        self.bpm_curve.setData(
            list(self.time_data), list(self.bpm_data)
        )  # Update the plot data
        self.bpm_label.setText(f"BPM: {average_bpm:.2f}")

    def calculate_average_ypm(self):
        elapsed_time_minutes = (
            datetime.now() - self.start_time
        ).total_seconds() / 60  # Convert elapsed time to minutes
        if elapsed_time_minutes > 0:
            return self.yawns / elapsed_time_minutes  # Calculate average YPM
        else:
            return 0

    def update_ypm_label(self):
        average_ypm = self.calculate_average_ypm()
        self.ypm_label.setText(f"YPM: {average_ypm:.2f}")
        # Make sure this part is reached and values are correct
        current_time = (datetime.now() - self.start_time).total_seconds() / 60
        self.ypm_data.append(average_ypm)
        self.ypm_curve.setData(list(self.time_data), list(self.ypm_data))

    @pyqtSlot()
    def update_yawn_count(self):
        print(self.yawns)
        self.yawns += 1
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
