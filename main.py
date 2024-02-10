import sys
from PyQt5.QtWidgets import QApplication
from MainMenuWindow import GUI  



if __name__ == "__main__":
    app = QApplication([])
    GUI = GUI()
    GUI.show()
    sys.exit(app.exec_())
