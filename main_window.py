# mainwindow.py
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QPushButton
from testing_button import Testing_Button
from testing_window import TestingWindow
from exit_button import exit_button


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Study Breaks")
        self.setGeometry(100, 100, 800, 600)  # x, y, width, height
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        self.layout = QVBoxLayout()
        self.centralWidget.setLayout(self.layout)


        self.exit_button = exit_button()
        self.layout.addWidget(self.exit_button)

        # Add the Testing_Button widget to the layout
        self.testing_button = Testing_Button()
        self.layout.addWidget(self.testing_button)

        # Connect the Testing_Button's signal to open the testing window
        self.testing_button.button.clicked.connect(self.open_testing_window)

    def open_testing_window(self):
        # Open the testing window when the button is clicked
        self.testing_window = TestingWindow()
        self.testing_window.show()



