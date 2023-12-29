from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton

class Testing_Button(QWidget):
    def __init__(self):
        super().__init__()

        # Main vertical layout
        main_layout = QVBoxLayout(self)

        # Add stretch to center vertically
        main_layout.addStretch(20)

        # Horizontal layout for the button
        button_layout = QHBoxLayout()
        button_layout.addStretch()  # Add stretch to center horizontally

        self.button = QPushButton("Testing", self)
        self.button.setFixedSize(150, 50)  # Set fixed size for the button
        button_layout.addWidget(self.button)

        button_layout.addStretch()  # Add stretch to center horizontally
        main_layout.addLayout(button_layout)  # Add horizontal layout to the main layout

        # Add stretch to center vertically
        main_layout.addStretch(0)
