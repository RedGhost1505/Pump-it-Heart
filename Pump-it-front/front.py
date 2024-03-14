from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QSpacerItem, QSizePolicy
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPalette, QColor


class SimpleWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setGeometry(320, 100, 900, 700)  # Set window position and size (x, y, width, height)
        self.setWindowTitle('Pump-it')  # Set window title

        # Set black background
        self.setStyleSheet("background-color: black;")

        label_text = '<font color="white">Pump-</font><font color="red">it</font>'
        label = QLabel(label_text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # Center-align the text

        # Set a specific font for the label
        font = QFont("Montserrat ExtraBold",70)
        font.setBold(False)
        label.setFont(font)
        

        # Create a QVBoxLayout to center the label in the window
        layout = QVBoxLayout(self)

        # Add spacer item at the top with a size of 10 pixels
        spacer = QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addItem(spacer)
        
        layout.addWidget(label, alignment=Qt.AlignmentFlag.AlignCenter)  # Center-align the label horizontally

        self.setLayout(layout)

        self.show()  # Display the window

if __name__ == '__main__':
    app = QApplication([])  # Create a PyQt6 application
    window = SimpleWindow()  # Create an instance of the SimpleWindow class
    app.exec()  # Start the application's event loop
