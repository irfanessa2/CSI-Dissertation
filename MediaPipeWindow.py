from collections import deque
import time
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QTimer, QThread,QDateTime
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QStackedWidget,QLabel,QVBoxLayout,QWidget,QRadioButton,QHBoxLayout,QProgressBar, QPushButton, QLineEdit, QComboBox, QMessageBox
import pyqtgraph as pg
from datetime import datetime
import numpy as np
import pandas as pd
from QWorker import CameraStream
import json
import os


#export CAMERA_SOURCE=Lady.mp4
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE")
assert(CAMERA_SOURCE)

# opencv needs the index to the OS video object
# try to convert to int
try:
    CAMERA_SOURCE = int(CAMERA_SOURCE)
except ValueError:
    # must have a path, so leave it
    pass


class MediaPipeWindow(QWidget):
    update = pyqtSignal()
    do_face = pyqtSignal(bool)
    do_blink = pyqtSignal(bool)
    do_yawn = pyqtSignal(bool)

    
    update_ear_threshold = pyqtSignal(float)

    def __init__(self, parent = None):
        super().__init__(parent)
        self.setWindowTitle("MediaPipe")
        self.resize(900, 480)
        self.blinks = 0
        self.yawn_mar_threshold = 10

        self.eye_fatigue_threshold = None
        
        self.blink_threshold_line = None
        
        self.yawn_fatigue_threshold = None
        self.yawn_threshold_line = None






        self.start_time = datetime.now()  # Initialize the start time
        self.ear = 0.0
        self.current_mar = None
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
                        
        self.bpm_curve = self.bpm_plot_widget.plot(pen="y")
        self.ypm_plot_widget = pg.PlotWidget(self)
        self.ypm_curve = self.ypm_plot_widget.plot(pen="r")

        self.image_label = QLabel(self)
        self.image_label_bot = QLabel(self)
        self.blink_count_label = QLabel(self)
        self.yawn_count_label = QLabel(self)
        self.cameras = [
            # Replace with your actual CameraStream
            CameraStream(CAMERA_SOURCE)
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


        self.calibration_timer = QLabel("UNCALIBRATED", self)
        self.eye_fatigue_threshold_label = QLabel("Eye Fatigue Threshold: Calculating...", self)
        self.yawn_fatigue_threshold_label = QLabel("Yawn Threshold: Calculating...", self)
        
        main_layout = QHBoxLayout()
        stats_layout = QVBoxLayout()
        frame_stats_layout = QHBoxLayout()
        yawn_stats_layout = QHBoxLayout()

        self.face_toggle = QRadioButton("Face Detection", self, autoExclusive=False)
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
        
        
        stats_layout.addWidget(self.calibration_timer)
        stats_layout.addWidget(self.eye_fatigue_threshold_label)
        stats_layout.addWidget(self.yawn_fatigue_threshold_label)


        stats_layout.addWidget(self.face_toggle)
        stats_layout.addWidget(self.blink_toggle)
        stats_layout.addWidget(self.yawn_toggle)



        profile_controls_layout = QHBoxLayout()


    # Add the profile loader dropdown to the layout
        self.profile_loader_dropdown = QComboBox(self)
        profile_controls_layout.addWidget(self.profile_loader_dropdown)

        # Add the "Load Selected Profile" button to the layout
        self.load_profile_button = QPushButton("Load Profile", self)
        self.load_profile_button.clicked.connect(self.load_profile)
        profile_controls_layout.addWidget(self.load_profile_button)

        # Add the "Delete Selected Profile" button to the layout
        self.delete_profile_button = QPushButton("Delete Profile", self)
        self.delete_profile_button.clicked.connect(self.delete_profile)
        profile_controls_layout.addWidget(self.delete_profile_button)

        # Assuming you have a layout called `stats_layout` where you want to insert these controls
        # Insert the horizontal layout into the stats_layout or wherever appropriate
        stats_layout.addLayout(profile_controls_layout)
        self.refreshProfiles()

        # Create a QHBoxLayout for the profile name input and the save button
        profile_save_layout = QHBoxLayout()

        # Create the profile name input QLineEdit
        self.profile_name_input = QLineEdit(self)
        self.profile_name_input.setPlaceholderText("Enter profile name")
        profile_save_layout.addWidget(self.profile_name_input)  # Add the input to the horizontal layout

        # Create the save profile button QPushButton
        self.save_profile_button = QPushButton("Save Profile", self)
        self.save_profile_button.clicked.connect(self.save_profile)
        profile_save_layout.addWidget(self.save_profile_button)  # Add the button to the horizontal layout

        # Add the QHBoxLayout (profile_save_layout) to the main QVBoxLayout (stats_layout)
        stats_layout.addLayout(profile_save_layout)

        

        self.export_data_button = QPushButton("Export Blink Data", self)
        self.export_data_button.clicked.connect(self.export_blink_data)
        stats_layout.addWidget(self.export_data_button)
        self.blinks_time_data = []


      
        

            
        self.calibrate_ear_button = QPushButton("Calibrate!", self)
        stats_layout.addWidget(self.calibrate_ear_button)  
        self.calibrate_ear_button.clicked.connect(self.start_calibration_process)








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

        self.blink_toggle.toggled.connect(
            lambda: self.do_blink.emit(self.blink_toggle.isChecked())
        )

        self.yawn_toggle.toggled.connect(
            lambda: self.do_yawn.emit(self.yawn_toggle.isChecked())
        )

        # [self.do_face.connect(cam.set_do_face) for cam in self.cameras]
        # [self.do_blink.connect(cam.set_do_blink) for cam in self.cameras]
        # [self.do_yawn.connect(cam.set_do_yawn) for cam in self.cameras]
        
        #self.do_face.connect(self.stream.set_do_face)
        #self.do_blink.connect(self.stream.set_do_blink)
        #self.do_yawn.connect(self.stream.set_do_yawn)
        #
        #self.do_face.connect(self.bot_cam.set_do_face)
        #self.do_blink.connect(self.bot_cam.set_do_blink)
        #self.do_yawn.connect(self.bot_cam.set_do_yawn)
        
        #self.cameras[0].looking_direction.connect(self.change_view)       #LOOKING DOWN HERE
        #self.stream.latency_signal.connect(self.latency_update)
        self.cameras[0].change_pixmap_signal.connect(self.update_image)
        #self.stream.ear_signal.connect(self.update_ear)
        #self.stream.eyes_closed_signal.connect(self.update_blink_count)
        #self.stream.eyes_closed_signal.connect(self.start_closed_eyes_timer)
        #self.stream.eyes_open_signal.connect(self.start_open_eyes_timer)
        self.cameras[0].signals.mar_signal.connect(self.update_mar)

        #self.bot_cam.latency_signal.connect(self.latency_update)
        #self.bot_cam.change_pixmap_signal.connect(self.update_image)
        #self.bot_cam.ear_signal.connect(self.update_ear)
        #self.bot_cam.eyes_closed_signal.connect(self.update_blink_count)
        #self.bot_cam.eyes_closed_signal.connect(self.start_closed_eyes_timer)
        #self.bot_cam.eyes_open_signal.connect(self.start_open_eyes_timer)
        #self.bot_cam.mar_signal.connect(self.update_mar)
        self.update.connect(self.update_values)
        # self.cameras[1].change_pixmap_signal.connect(self.update_image_bot)
        self.update_values()
        self.connect_signals(self.cameras[0])
        self.threads = set() # prevent anonymous threads from being garbage collected (very oddgy if your resource managements its good; it's not :p)
        [self.start_camera_thread(cam) for cam in self.cameras]
        



    def collect_blink_data(self):
            current_time = (datetime.now() - self.start_time).total_seconds() / 60  # Current time in minutes
            self.blinks_time_data.append((current_time, self.blinks))
        
    # Method to export blink data to Excel
    def export_blink_data(self):
        # Convert the blink data to a DataFrame
        df = pd.DataFrame(self.blinks_time_data, columns=['Time (Minutes)', 'Blinks'])
        
        # Create a Pandas Excel writer using openpyxl as the engine
        with pd.ExcelWriter('blink_data.xlsx', engine='openpyxl') as writer:
            # Convert the DataFrame to an Excel object
            df.to_excel(writer, sheet_name='Sheet1', index=False)
        
        print("Blink data exported successfully.")


        
        
        
    def start_camera_thread(self, cam):
        thread = QThread()
        self.connect_signals(cam) # connect camera signals
        self.threads.add(thread) # store a reference
        cam.moveToThread(thread)
        thread.started.connect(cam.run)
        thread.finished.connect(cam.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self.update_ear_threshold.connect(cam.set_ear_threashold)
        thread.start()

    def connect_signals(self, camera):
        cam_sig = camera.signals
        cam_sig.latency_signal.connect(self.latency_update)
        cam_sig.ear_signal.connect(self.update_ear)
        cam_sig.mar_signal.connect(self.update_mar)
        cam_sig.tilt_signal.connect(self.update_tilt)
        cam_sig.eyes_closed_signal.connect(self.update_blink_count)
        cam_sig.eyes_closed_signal.connect(self.start_closed_eyes_timer)
        cam_sig.eyes_open_signal.connect(self.start_open_eyes_timer)

        self.do_face.connect(camera.set_do_face)
        self.do_blink.connect(camera.set_do_blink)
        self.do_yawn.connect(camera.set_do_yawn)

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
        self.current_mar = val
        self.mar.setText(f"Mouth AR: {val:.2f}")
        if val > self.yawn_mar_threshold:
            yawn_type = "YAWNING"
            if not self.is_yawning:  # Check if transition to yawning state
                self.is_yawning = True
                self.update_yawn_count()
        elif val < self.yawn_mar_threshold:
            yawn_type = "CLOSED"
            self.is_yawning = False  # Reset yawn state when mouth is closed
        else:
            # Removed the  handling of "TALKING" state
            # may choose to not set yawn_type here or handle it differently
            self.is_yawning = False  # Ensure yawn state is reset for other conditions
        if 'yawn_type' in locals():
            self.yawn_type.setText(yawn_type)
        # Only update yawn_type if it's defined (i.e., not talking


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
        self.blinks += 0.5
        
        if self.blinks % 1 == 0:  # Checks if `self.blinks` is a whole number
            self.collect_blink_data()  # Now this gets called once per blink
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
            if elapsed_time.total_seconds() >= 20:  # Assuming 10 seconds for example
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
            #print(self.eye_fatigue_threshold)
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
        # print(self.camera_stream_instance.ear_threshold)
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
            #print(self.yawn_fatigue_threshold)
            self.yawn_fatigue_threshold_label.setText(f"Yawn Fatigue Threshold: {self.yawn_fatigue_threshold:.2f} YPM")      
                      

    def update_yawn_threshold_label(self, average_ypm):
        if self.yawn_fatigue_threshold is not None:

            if average_ypm > self.yawn_fatigue_threshold:
                self.yawn_fatigue_threshold_label.setText("Yawn Fatigue")
                self.yawn_fatigue_threshold_label.setStyleSheet("color: red")
            else:
                self.yawn_fatigue_threshold_label.setText(f"Yawn Threshold ={self.yawn_fatigue_threshold: .2f} YPM: OK")
                self.yawn_fatigue_threshold_label.setStyleSheet("color: green")



    

        
    def getmin_max_ear(self):
        self.ear_values = []  
        self.calibration_start_time = QDateTime.currentDateTime()  # Capture start time

        def collect_ear():
            self.ear_values.append(self.ear)
            # Calculate elapsed time in seconds
            elapsed_time = self.calibration_start_time.secsTo(QDateTime.currentDateTime())
            remaining_time = max(0, 20 - elapsed_time)  # Ensure remaining time doesn't go below 0
            if remaining_time > 0:
                self.calibration_timer.setText(f"Calibrating... {remaining_time}s")
                self.calibration_timer.setStyleSheet("color: red;")
            else:
                # Stop the timer and call finish_ear_collection if not already called
                self.collect_ear_timer.stop()
                self.finish_ear_collection()

        # Initialize the QTimer instance for collecting EAR values at regular intervals
        self.collect_ear_timer = QTimer(self)
        self.collect_ear_timer.timeout.connect(collect_ear)
        self.collect_ear_timer.start(100)  # Collecting EAR values every 100 ms.



    def finish_ear_collection(self):
        # Stop the timer and process collected EAR values.
        self.collect_ear_timer.stop()
        if self.ear_values:
            self.cameras[0].min_ear = min(self.ear_values)
            self.cameras[0].max_ear = max(self.ear_values)
            self.cameras[0].ear_threshold = (self.cameras[0].min_ear + ((self.cameras[0].max_ear - self.cameras[0].min_ear)) * 0.3)

            self.update_ear_threshold.emit(self.cameras[0].ear_threshold)
            # Update the calibration timer label to show "Calibrated" in green
            self.calibration_timer.setText("Calibrated!")
            self.calibration_timer.setStyleSheet("color: green;")
            
            print(f"Min EAR: {self.cameras[0].min_ear}, Max EAR: {self.cameras[0].max_ear}")
            print(f"EAR Threshold: {self.cameras[0].ear_threshold}")
            print(f"EAR Threshold: {self.cameras[0].ear_threshold}")

            
        else:
            # If no EAR values were collected, indicate calibration failed
            self.calibration_timer.setText("Calibration Failed")
            self.calibration_timer.setStyleSheet("color: red;")
            print("No EAR values were collected.")

       
           
           
           
           
           
           
           
           
           
           


    def getmin_max_mar(self):
        self.mar_values = []  
        self.calibration_start_time = QDateTime.currentDateTime()  # Capture start time

        def collect_mar():
            self.mar_values.append(self.current_mar)            # Calculate elapsed time in seconds
            elapsed_time = self.calibration_start_time.secsTo(QDateTime.currentDateTime())
            remaining_time = max(0, 20 - elapsed_time)  # Ensure remaining time doesn't go below 0
            if remaining_time < 0:

                # Stop the timer and call finish_mar_collection if not already called
                self.collect_mar_timer.stop()
                self.finish_mar_collection()

        # Initialize the QTimer instance for collecting EAR values at regular intervals
        self.collect_mar_timer = QTimer(self)
        self.collect_mar_timer.timeout.connect(collect_mar)
        self.collect_mar_timer.start(100)  # Collecting EAR values every 100 ms.



    def finish_mar_collection(self):
        # Stop the timer and process collected MAR values.
        self.collect_mar_timer.stop()
        if self.mar_values:  # Make sure there are values collected
            self.min_mar = min(self.mar_values)
            self.max_mar = max(self.mar_values)
            self.yawn_mar_threshold = (self.min_mar + (self.max_mar - self.min_mar) * 0.3)


            # Here you might want to do something with self.mar_threshold,
            # like emitting it with a signal or updating some UI element

            print(f"Min MAR: {self.min_mar}, Max MAR: {self.max_mar}")

        else:
            print("No MAR values were collected.")

           
           
           
           
    def start_calibration_process(self):
        if self.current_mar is not None and self.ear != 0.0:
            self.getmin_max_ear()
            self.getmin_max_mar()
           
           

    def save_profile(self):
        profile_name = self.profile_name_input.text().strip()
        if not profile_name:  # Check if the profile name is not empty
            print("Please enter a profile name.")
            return
        
        profile_dir = 'profiles'  # Directory name
        if not os.path.exists(profile_dir):
            os.makedirs(profile_dir)  # Create the directory if it doesn't exist
        
        profile_data = {
            "name": profile_name,
            "ear_threshold": self.cameras[0].ear_threshold,
            "yawn_mar_threshold": self.yawn_mar_threshold,
        }
        
        file_path = os.path.join(profile_dir, f"{profile_name}.json")  # Adjust file path
        with open(file_path, "w") as file:
            json.dump(profile_data, file, indent=4)
        
        print(f"Profile '{profile_name}' saved successfully.")
        self.refreshProfiles()  # Refresh the profiles list if this function is defined

     
     
     
    def load_profile(self):
        selected_profile = self.profile_loader_dropdown.currentText()
        if not selected_profile:
            print("No profile selected.")
            return
        
        profile_path = os.path.join('profiles', selected_profile)  # Adjust if you use a different directory
        try:
            with open(profile_path, 'r') as file:
                profile_data = json.load(file)
                self.cameras[0].ear_threshold = profile_data.get('ear_threshold', 0.0)  # Provide default value
                self.yawn_mar_threshold = profile_data.get('yawn_mar_threshold', 1.0)  # Provide default value
                # Update any UI or variables that depend on these values
                print(f"Loaded profile '{selected_profile}' successfully.")
                self.calibration_timer.setText("Profile loaded. Calibrated!")
                self.calibration_timer.setStyleSheet("color: green;")
                self.update_ear_threshold.emit(self.cameras[0].ear_threshold)

                # self.yawn_mar_threshold.emit(self.yawn_mar_threshold)

        except Exception as e:
            print(f"Failed to load profile '{selected_profile}': {e}")
            
            
    def refreshProfiles(self):
        self.profile_loader_dropdown.clear()
        profile_dir = 'profiles'  # or '.' for the current directory
        profile_files = [f for f in os.listdir(profile_dir) if f.endswith('.json')]
        self.profile_loader_dropdown.addItems(profile_files)


    def delete_profile(self):
        selected_profile = self.profile_loader_dropdown.currentText()
        if not selected_profile:
            print("No profile selected for deletion.")
            return

        confirmation = QMessageBox.question(self, "Confirm Deletion", f"Are you sure you want to delete the profile '{selected_profile}'?", QMessageBox.Yes | QMessageBox.No)
        if confirmation == QMessageBox.Yes:
            profile_path = os.path.join('profiles', selected_profile)
            try:
                os.remove(profile_path)
                print(f"Profile '{selected_profile}' deleted successfully.")
                self.refreshProfiles()  # Refresh the profiles list
            except Exception as e:
                print(f"Failed to delete profile '{selected_profile}': {e}")
        
    @pyqtSlot(float)
    def update_ear(self, val):
        self.ear = val
        self.update.emit()




