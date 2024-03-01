import sys
from PyQt5.QtWidgets import QApplication
from MainMenuWindow import GUI  
import subprocess
import os

UDEV_SCRIPT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),'udev_rules.sh'))


if __name__ == "__main__":
    subprocess.Popen([
        'bash',
        UDEV_SCRIPT_PATH
        ])
    app = QApplication(sys.argv)
    GUI = GUI()
    GUI.show()
    sys.exit(app.exec_())
