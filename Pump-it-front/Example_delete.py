import sys
import cv2
from PyQt6.QtCore import pyqtSignal, QThread, Qt
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt6.QtGui import QImage, QPixmap

class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)

    def __init__(self, camera_index):
        super().__init__()
        self.camera_index = camera_index
        self.running = True

    def run(self):
        # Asegúrate de que el índice de la cámara es accesible y funcional
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"Unable to access camera with index {self.camera_index}")
            self.running = False
        while self.running:
            ret, cv_img = cap.read()
            if ret:
                rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                p = convert_to_Qt_format.scaled(640, 480, aspectMode=Qt.AspectRatioMode.KeepAspectRatio)
                self.change_pixmap_signal.emit(p)
        cap.release()

    def stop(self):
        self.running = False
        self.quit()
        self.wait()

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Camera Viewer")
        layout = QVBoxLayout()

        self.image_label1 = QLabel(self)
        self.image_label2 = QLabel(self)
        layout.addWidget(self.image_label1)
        layout.addWidget(self.image_label2)

        self.setLayout(layout)
        
        # Initialize camera threads
        self.thread1 = CameraThread(0)  # Assuming first camera index
        self.thread1.change_pixmap_signal.connect(self.update_image1)
        self.thread1.start()

        self.thread2 = CameraThread(1)  # Assuming second camera index
        self.thread2.change_pixmap_signal.connect(self.update_image2)
        self.thread2.start()

    def update_image1(self, image):
        self.image_label1.setPixmap(QPixmap.fromImage(image))

    def update_image2(self, image):
        self.image_label2.setPixmap(QPixmap.fromImage(image))

    def closeEvent(self, event):
        self.thread1.stop()
        self.thread2.stop()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = App()
    main_window.show()
    sys.exit(app.exec())
