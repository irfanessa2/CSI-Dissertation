import cv2
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap

class CameraStream(QThread):
    change_pixmap_signal = pyqtSignal(QImage)

    def __init__(self, src=0, parent=None):
        super().__init__(parent)
        self.src = src
        self.stopped = False

    def run(self):
        cap = cv2.VideoCapture(self.src)
        if not cap.isOpened():
            raise IOError("Cannot open camera stream: {}".format(self.src))

        while not self.stopped:
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
                self.change_pixmap_signal.emit(p)
            else:
                self.stopped = True

        cap.release()

    def stop(self):
        self.stopped = True
        self.wait()

class TestingWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Testing Window")
        self.resize(640, 480)
        self.layout = QVBoxLayout(self)
        self.image_label = QLabel(self)
        self.layout.addWidget(self.image_label)
        self.setLayout(self.layout)

        self.stream = CameraStream(0, self)
        self.stream.change_pixmap_signal.connect(self.update_image)
        self.stream.start()

    @pyqtSlot(QImage)
    def update_image(self, image):
        self.image_label.setPixmap(QPixmap.fromImage(image))

    def closeEvent(self, event):
        self.stream.stop()
        super().closeEvent(event)
