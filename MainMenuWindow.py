from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QHBoxLayout)
from PyQt5.QtCore import Qt
from MediaPipeWindow import MediaPipeWindow  # Ensure this import matches your file structure

class MediaPipe_Button(QWidget):
    def __init__(self):
        super().__init__()

        main_layout = QVBoxLayout(self)
        main_layout.addStretch(1)

        button_layout = QHBoxLayout()
        button_layout.addStretch(1)

        self.button = QPushButton("MediaPipe", self)
        self.button.setFixedSize(150, 50)
        button_layout.addWidget(self.button)

        button_layout.addStretch(1)
        main_layout.addLayout(button_layout)
        main_layout.addStretch(0)

class ExitButton(QWidget):
    def __init__(self):
        super().__init__()

        main_layout = QVBoxLayout(self)
        main_layout.addStretch(1)

        button_layout = QHBoxLayout()
        button_layout.addStretch(1)

        self.button = QPushButton("Exit", self)
        self.button.setFixedSize(150, 50)
        button_layout.addWidget(self.button)

        button_layout.addStretch(1)
        main_layout.addLayout(button_layout)
        main_layout.addStretch(20)

        self.button.clicked.connect(QApplication.instance().quit)

class GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fatigue Detection")
        self.setGeometry(100, 100, 800, 600)

        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        self.layout = QVBoxLayout(self.centralWidget)

        self.MediaPipe_Button = MediaPipe_Button()
        self.layout.addWidget(self.MediaPipe_Button)
        self.MediaPipe_Button.button.clicked.connect(self.open_MediaPipe_Window)

        self.exit_button = ExitButton()
        self.layout.addWidget(self.exit_button)

    def open_MediaPipe_Window(self):
        self.MediaPipe_Window = MediaPipeWindow()
        self.MediaPipe_Window.show()


