import time
import cv2
import mediapipe as mp
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot, Qt, QTimer
from PyQt5.QtGui import QImage
import numpy as np
from threading import Thread

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE_INDICES = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398,]
RIGHT_EYE_INDICES = [133,153,154,155,145,144,163,7,33,246,161,160,159,158,157,173,]
UPPER_LIP_INDICES = [13, 312, 311, 310, 415, 308, 324, 318, 402, 317]
LOWER_LIP_INDICES = [14, 87, 178, 88, 95, 78, 191, 80, 81, 82]
NOSE_TIP_INDEX = 1


RIGHT_EYE = [159,145,157,25]
LEFT_EYE = [386,374,414,263]
MOUTH = [13,14,78,308]


class CameraSignals(QObject):
    ear_signal = pyqtSignal(float)
    mar_signal = pyqtSignal(float)
    tilt_signal = pyqtSignal(float)
    eyes_closed_signal = pyqtSignal()
    eyes_open_signal = pyqtSignal()
    latency_signal = pyqtSignal(object)


class CameraStream(QObject):
    change_pixmap_signal = pyqtSignal(QImage)
    looking_direction = pyqtSignal(int)

    def __init__(self, src=0) :
        super().__init__()
        self.min_ear = 0
        self.max_ear = 0
        self.ear_threshold = 0


        self.src = src
        self.signals = CameraSignals()
        self.stopped = False
        self.face_mesh = mp_face_mesh.FaceMesh()
        self.blinked = False
        self.opened = False
        self.face_feature_detection_thread = Thread()
        self.face_results = None
        self.do_face = False
        self.do_blink = False
        self.do_yawn = False
        self.printed_once = None
        self.cap = cv2.VideoCapture(src)

    @pyqtSlot(float)
    def set_ear_threashold(self, ear):
        self.ear_threshold = ear



    @pyqtSlot(bool)
    def set_do_face(self, val):
        self.do_face = val

    @pyqtSlot(bool)
    def set_do_yawn(self, val):
        self.do_yawn = val

    @pyqtSlot(bool)
    def set_do_blink(self, val):
        self.do_blink = val





    def face_feature_detection_worker(self, frame):
        # Convert to RGB once
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.face_results = self.face_mesh.process(frame_rgb)
        if self.face_results.multi_face_landmarks:
            for face_landmarks in self.face_results.multi_face_landmarks:
                # Print for debugging
                # Looking down detection
                if CameraStream.is_looking_down(face_landmarks):
                    if not self.printed_once:
                        print("Looking down detected")  # This should now print if the method is reached
                        self.printed_once = True
                    self.looking_direction.emit(1)
                # Additional print for debugging
                else:
                    if self.printed_once:
                        print("Looking down NOT detected")
                        self.looking_direction.emit(0)
                        self.printed_once = False


    @staticmethod
    def is_looking_down(face_landmarks):
        # Extract the eye and nose landmarks
        nose_tip = np.array([face_landmarks.landmark[NOSE_TIP_INDEX].x, face_landmarks.landmark[NOSE_TIP_INDEX].y])
        left_eye = CameraStream.get_feature_landmarks(face_landmarks.landmark, LEFT_EYE_INDICES)
        right_eye = CameraStream.get_feature_landmarks(face_landmarks.landmark, RIGHT_EYE_INDICES)

        # Calculate average eye position
        avg_eye_y = np.mean([ey[1] for ey in left_eye + right_eye])

        # Define a threshold for looking down, appropriate for normalized coordinates
        LOOK_DOWN_THRESHOLD = 0.103 # Adjust based on testing  0.1

        # Check if average eye Y position is less than the nose tip Y position by the threshold
        looking_down = avg_eye_y < nose_tip[1] - LOOK_DOWN_THRESHOLD

        # Debugging prints
        #print(f"Average Eye Y: {avg_eye_y}, Nose Tip Y: {nose_tip[1]}, Threshold: {LOOK_DOWN_THRESHOLD}, Looking Down: {looking_down}")

        return looking_down

                        


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
                    left_ear, tilt_l = self.calculate_ar(self.get_feature_landmarks(face_landmarks.landmark, LEFT_EYE))
                    right_ear, tilt_r = self.calculate_ar(self.get_feature_landmarks(face_landmarks.landmark, RIGHT_EYE))
                    self.signals.tilt_signal.emit((tilt_l+tilt_r)/2)
                    ear = (left_ear + right_ear) / 2.0
                    self.signals.ear_signal.emit(ear)

                    feature_points = feature_points + right_eye + left_eye
                    

                    if self.do_blink:
                        #if ear < self.ear_threshold:
                        #print (self.ear_threshold)                    
                            if ear < self.ear_threshold:
                                if not self.blinked:
                                    self.blinked = True
                                    self.opened = False
                                    self.signals.eyes_closed_signal.emit()
                            elif not self.opened:
                                self.opened = True
                                self.blinked = False
                                self.signals.eyes_open_signal.emit()




                if self.do_yawn:
                    mouth = self.get_feature_landmarks(
                        face_landmarks.landmark, UPPER_LIP_INDICES + LOWER_LIP_INDICES
                    )
                    mar, tilt = self.calculate_ar(self.get_feature_landmarks(face_landmarks.landmark, MOUTH))
                    self.signals.mar_signal.emit(mar)

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
        """ Make this thread async, fire of a read event every 1000//30 ms
            this gives the event loop time to process signals
            PS: there is no resource management, make sure to run in debugger
                or kill -9 any ps aux | grep python"""
        self.timer = QTimer()
        self.timer.setInterval(1000//30) # 30fps
        self.timer.timeout.connect(self._run)
        self.timer.start()
        
    

    def _run(self):
        self.signals.latency_signal.emit(time.time_ns())
        ret, frame = self.cap.read()
        if ret:
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
    
    @pyqtSlot()
    def close(self):
        self.timer.stop()
        self.timer.deleteLater()
        self.cap.release()
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
    def calculate_ar(points):
        #[ [topx,topy], [bottomx,bottomy], [leftx,lefty], [rightx,righty] ] 
        
        top, bottom, left, right = points

        dx = np.linalg.norm(top-bottom)
        dy = np.linalg.norm(left-right)      #d = difference, eucledian
        
        if dy > 0:
            ar = dx/dy
            tilt = np.arctan((left[1]-right[1])/(left[0]-right[0]))
        else:
            ar = -1
            tilt = -1    
        
        return (ar, tilt)
        
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


