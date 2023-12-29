# main.py
import sys
from PyQt5.QtWidgets import QApplication
from main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
