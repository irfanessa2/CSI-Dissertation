from collections import deque
import time
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QStackedWidget,QLabel,QVBoxLayout,QWidget,QRadioButton,QHBoxLayout,QProgressBar
import pyqtgraph as pg
from datetime import datetime
import numpy as np
from QWorker import CameraStream


class MediaPipeWindow(QWidget):
    update = pyqtSignal()
    do_face = pyqtSignal(bool)
    do_hands = pyqtSignal(bool)
    do_blink = pyqtSignal(bool)
    do_yawn = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MediaPipe")
        
        self.resize(900, 480)
        self.blinks = 0
        
        self.eye_fatigue_threshold = None
        self.blink_threshold_line = None
        
        self.yawn_fatigue_threshold = None
        self.yawn_threshold_line = None


        self.start_time = datetime.now()  # Initialize the start time
        self.ear = 0.0
        self.yawns = 0
        self.frame_ts = 0
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
        
        self.time_data2 = deque(
            maxlen=300
        )  # Store up to the last 300 data points for time
        self.ypm_data = deque(
            maxlen=300
        )  # Store up to the last 300 data points for YPM

        # Initialize plot widget
        self.bpm_plot_widget = pg.PlotWidget(self)
        self.bpm_curve = self.bpm_plot_widget.plot(pen="y")
        self.ypm_plot_widget = pg.PlotWidget(self)
        self.ypm_curve = self.ypm_plot_widget.plot(pen="r")

        self.image_label = QLabel(self)
        self.image_label_bot = QLabel(self)
        self.blink_count_label = QLabel(self)
        self.yawn_count_label = QLabel(self)
        self.cameras = [
            CameraStream(0, self),  # Replace with your actual CameraStream
            CameraStream(2, self)
            ]
        self.fps_label = QLabel("FPS: 0", self)
        self.bpm_label = QLabel("BPM: 0", self)  # Use for displaying text BPM
        self.ypm_label = QLabel("YPM: 0", self)  # Use for displaying text YPM
        self.tilt_label = QLabel("Tilt: 0", self)  # Use for displaying text YPM

        self.mar = QLabel("Mouth AR:", self)
        self.yawn_type = QLabel(self)
        self.latency_label = QLabel("Latency: 0", self)
        self.asleep_label = QLabel("ASLEEP", self)
        self.asleep_label.setStyleSheet("color:red")
        self.asleep_label.hide()

        self.eye_fatigue_threshold_label = QLabel("Eye Fatigue Threshold: Calculating...", self)
        self.yawn_fatigue_threshold_label = QLabel("Yawn Threshold: Calculating...", self)
        
        main_layout = QHBoxLayout()
        stats_layout = QVBoxLayout()
        frame_stats_layout = QHBoxLayout()
        yawn_stats_layout = QHBoxLayout()

        self.face_toggle = QRadioButton("Face Detection", self, autoExclusive=False)
        self.hands_toggle = QRadioButton("Hands Detection", self, autoExclusive=False)
        self.blink_toggle = QRadioButton("Blink Detection", self, autoExclusive=False)
        self.yawn_toggle = QRadioButton("Yawn Detection", self, autoExclusive=False)

        self.is_yawning = False
        self.opened_eyes_triggered = False

        self.sleep_bar = QProgressBar(self)
        self.sleep_bar.setMaximum(3 * 1000)
        self.sleep_bar.setValue(0)

        yawn_stats_layout.addWidget(self.mar)
        yawn_stats_layout.addWidget(self.yawn_type)

        frame_stats_layout.addWidget(self.fps_label)
        frame_stats_layout.addWidget(self.latency_label)
        frame_stats_layout.addWidget(self.bpm_label)
        frame_stats_layout.addWidget(self.ypm_label)
        frame_stats_layout.addWidget(self.tilt_label)
        
        stats_layout.addWidget(self.eye_fatigue_threshold_label)
        stats_layout.addWidget(self.yawn_fatigue_threshold_label)


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

        self. sw = QStackedWidget(self)
        self.sw.addWidget(self.image_label)
        self.sw.addWidget(self.image_label_bot)
        main_layout.addWidget(self.sw)
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

        [self.do_face.connect(cam.set_do_face) for cam in self.cameras]
        [self.do_hands.connect(cam.set_do_hands) for cam in self.cameras]
        [self.do_blink.connect(cam.set_do_blink) for cam in self.cameras]
        [self.do_yawn.connect(cam.set_do_yawn) for cam in self.cameras]
        #self.do_face.connect(self.stream.set_do_face)
        #self.do_hands.connect(self.stream.set_do_hands)
        #self.do_blink.connect(self.stream.set_do_blink)
        #self.do_yawn.connect(self.stream.set_do_yawn)
        #
        #self.do_face.connect(self.bot_cam.set_do_face)
        #self.do_hands.connect(self.bot_cam.set_do_hands)
        #self.do_blink.connect(self.bot_cam.set_do_blink)
        #self.do_yawn.connect(self.bot_cam.set_do_yawn)
        
        self.cameras[0].looking_direction.connect(self.change_view)       #LOOKING DOWN HERE
        #self.stream.latency_signal.connect(self.latency_update)
        self.cameras[0].change_pixmap_signal.connect(self.update_image)
        #self.stream.ear_signal.connect(self.update_ear)
        #self.stream.eyes_closed_signal.connect(self.update_blink_count)
        #self.stream.eyes_closed_signal.connect(self.start_closed_eyes_timer)
        #self.stream.eyes_open_signal.connect(self.start_open_eyes_timer)
        #self.stream.mar_signal.connect(self.update_mar)

        #self.bot_cam.latency_signal.connect(self.latency_update)
        #self.bot_cam.change_pixmap_signal.connect(self.update_image)
        #self.bot_cam.ear_signal.connect(self.update_ear)
        #self.bot_cam.eyes_closed_signal.connect(self.update_blink_count)
        #self.bot_cam.eyes_closed_signal.connect(self.start_closed_eyes_timer)
        #self.bot_cam.eyes_open_signal.connect(self.start_open_eyes_timer)
        #self.bot_cam.mar_signal.connect(self.update_mar)
        self.update.connect(self.update_values)
        self.cameras[1].change_pixmap_signal.connect(self.update_image_bot)
        [cam.start() for cam in self.cameras]
        self.update_values()
        self.connect_signals(self.cameras[0])

    def connect_signals(self, camera):
        cam_sig = camera.signals
        cam_sig.latency_signal.connect(self.latency_update)
        cam_sig.ear_signal.connect(self.update_ear)
        cam_sig.mar_signal.connect(self.update_mar)
        cam_sig.tilt_signal.connect(self.update_tilt)
        cam_sig.eyes_closed_signal.connect(self.update_blink_count)
        cam_sig.eyes_closed_signal.connect(self.start_closed_eyes_timer)
        cam_sig.eyes_open_signal.connect(self.start_open_eyes_timer)

    def disconnect_signals(self, camera):
        for attr in dir(camera.signals):
            if attr.endswith('signal'):
                getattr(camera.signals, attr).disconnect() 
                
    @pyqtSlot(float)
    def update_tilt(self, a):
        #angle comes in in radians, convert to degrees
        def rad2deg(rad):
            return round((180/np.pi)*rad)
        text = str(rad2deg(a))+u'\u00b0'
        self.tilt_label.setText(text)

    @pyqtSlot(int)
    def change_view(self, view):
        if self.sw.currentIndex() != view:
            self.sw.setCurrentIndex(view)
            self.disconnect_signals(self.cameras[view-1])
            self.connect_signals(self.cameras[view])

    @pyqtSlot(float)
    def update_mar(self, val):
        self.mar.setText(f"Mouth AR: {val:.2f}")
        if val > 0.8:
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
        if self.opened_eyes_triggered:
            self.stop_all_timers()
            self.eyes_closed_timer.start()
            self.opened_eyes_triggered = False

    def start_open_eyes_timer(self):
        if not self.opened_eyes_triggered:
            self.stop_all_timers()
            self.eyes_opened_timer.start()
            self.opened_eyes_triggered = True

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

    @pyqtSlot(QImage)
    def update_image_bot(self, image):
        frame_ts = time.time_ns()
        self.fps.append(frame_ts - self._last_frame_ts)
        self._last_frame_ts = frame_ts
        self.latency.append(time.time_ns() - self.frame_ts)
        self.latency_label.setText(f"Latency: {np.mean(self.latency)/1e6:.2f}ms")
        self.fps_label.setText(f"FPS: {1e9/np.mean(self.fps):.2f}")
        self.update_bpm_label()
        self.update_ypm_label()
        self.image_label_bot.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot()
    def update_values(self):
        self.blink_count_label.setText(f"Blinks: {self.blinks}, EAR: {self.ear:.2f}")

    @pyqtSlot()
    def update_blink_count(self):
        self.blinks += 1
        if self.eye_fatigue_threshold is None:  
            self.calculate_eye_fatigue_threshold()
        self.update.emit()

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
        
        self.update_blink_threshold_label(average_bpm)
        
             
    def calculate_eye_fatigue_threshold(self):
        if self.eye_fatigue_threshold is None:
            elapsed_time = datetime.now() - self.start_time
            if elapsed_time.total_seconds() >= 10:  # Assuming 10 seconds for example
                self.eye_fatigue_threshold = self.calculate_average_bpm() * 1.5
                self.add_blink_threshold_line()


    def add_blink_threshold_line(self):
        # Add the threshold line only if it hasn't been added yet
        if self.blink_threshold_line is None:
            self.blink_threshold_line = pg.InfiniteLine(
                angle=0,  # Horizontal line
                pos=self.eye_fatigue_threshold,  # Position at the eye fatigue threshold value
                pen=pg.mkPen('w', width=2)  # White line
            )
            self.bpm_plot_widget.addItem(self.blink_threshold_line)
            print(self.eye_fatigue_threshold)
            self.eye_fatigue_threshold_label.setText(f"Eye Fatigue Threshold: {self.eye_fatigue_threshold:.2f} BPM")


    def update_blink_threshold_label(self, average_bpm):
        if self.eye_fatigue_threshold is not None:

            if average_bpm > self.eye_fatigue_threshold:
                self.eye_fatigue_threshold_label.setText("Eye Fatigue")
                self.eye_fatigue_threshold_label.setStyleSheet("color: red")
            else:
                self.eye_fatigue_threshold_label.setText(f"Blink Threshold ={self.eye_fatigue_threshold: .2f} BPM: OK")
                self.eye_fatigue_threshold_label.setStyleSheet("color: green")


    def calculate_average_ypm(self):
        elapsed_time_minutes = (
            datetime.now() - self.start_time
        ).total_seconds() / 60  # Convert elapsed time to minutes
        if elapsed_time_minutes > 0:
            return self.yawns / elapsed_time_minutes  # Calculate average YPM
        else:
            return 0

    def update_ypm_label(self):
        # Call this method at a regular interval to update the plot
        average_ypm = self.calculate_average_ypm()
        current_time = (
            datetime.now() - self.start_time
        ).total_seconds() / 60  # Current time in minutes
        self.time_data2.append(current_time)
        self.ypm_data.append(average_ypm)
        self.ypm_curve.setData(
            list(self.time_data2), list(self.ypm_data)
        )  # Update the plot data
        self.ypm_label.setText(f"YPM: {average_ypm:.2f}")   
        self.update_yawn_threshold_label(average_ypm)
        
            
    @pyqtSlot()
    def update_yawn_count(self):
        print(self.yawns)
        self.yawns += 1
        if self.yawn_fatigue_threshold is None:  
            self.calculate_yawn_fatigue_threshold()
        self.update.emit()
        
        
    def calculate_yawn_fatigue_threshold(self):
        if self.yawn_fatigue_threshold is None:
            elapsed_time = datetime.now() - self.start_time
            if elapsed_time.total_seconds() >= 10:  # Assuming 10 seconds for example
                self.yawn_fatigue_threshold = self.calculate_average_ypm() * 1.5
                self.add_yawn_threshold_line()        
                
    def add_yawn_threshold_line(self):
        # Add the threshold line only if it hasn't been added yet
        if self.yawn_threshold_line is None:
            self.yawn_threshold_line = pg.InfiniteLine(
                angle=0,  # Horizontal line
                pos=self.yawn_fatigue_threshold,  # Position at the eye fatigue threshold value
                pen=pg.mkPen('w', width=2)  # White line
            )
            self.ypm_plot_widget.addItem(self.yawn_threshold_line)
            print(self.yawn_fatigue_threshold)
            self.yawn_fatigue_threshold_label.setText(f"Yawn Fatigue Threshold: {self.yawn_fatigue_threshold:.2f} YPM")      
                      

    def update_yawn_threshold_label(self, average_ypm):
        if self.yawn_fatigue_threshold is not None:

            if average_ypm > self.yawn_fatigue_threshold:
                self.yawn_fatigue_threshold_label.setText("Yawn Fatigue")
                self.yawn_fatigue_threshold_label.setStyleSheet("color: red")
            else:
                self.yawn_fatigue_threshold_label.setText(f"Yawn Threshold ={self.yawn_fatigue_threshold: .2f} YPM: OK")
                self.yawn_fatigue_threshold_label.setStyleSheet("color: green")
                
                
    @pyqtSlot(float)
    def update_ear(self, val):
        self.ear = val
        self.update.emit()




