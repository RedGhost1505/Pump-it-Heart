from interfaz import *
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import random

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)
        # Configuracion Firebase
        cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred)
        # Configuración de BD (Firestore)
        db = firestore.client()
        self.clima_ref = db.collection('clima')
        self.actuales_ref = db.collection('actuales').document('TN1hIgo15SNU98M874tE')

        # Logica
        self.pushButton.clicked.connect(self.send_data)
        self.pushButton_2.clicked.connect(self.read_data)
    
    def send_data(self):
        t = self.get_data_sensor(10,40)
        h = self.get_data_sensor(20,60)
        p = self.get_data_sensor(800,1500)

        self.db_create(t, h, p)
        self.db_update_actuales(t, h, p, True)

    def get_data_sensor(self,min,max):
        return random.randrange(min, max)
        
    def db_create(self,t,h,p):
        data = {
            'temp': t,
            'hum': h,
            'pres': p,
            'timestamp': datetime.now() 
        }
        self.clima_ref.document().set(data)

    def db_update_actuales(self, t, h ,p, l):
        data = {
            'temp': t,
            'hum': h,
            'pres': p,
            'led': l
        }
        self.actuales_ref.update(data)

    def read_data(self):
        data = self.actuales_ref.get()
        data = data.to_dict()
        print(data)

        self.lcdNumber.display(data['temp'])
        self.lcdNumber_2.display(data['hum'])
        self.lcdNumber_3.display(data['pres'])

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec()