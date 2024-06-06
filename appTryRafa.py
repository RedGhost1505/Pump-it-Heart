import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()
clima_ref = db.collection('clima')
actuales_ref = db.collection('actuales').document('TN1hIgo15SNU98M874tE')


def create_data(t,h,p):
    data = {
        'temp': t,
        'hum': h,
        'pres': p,
        'timestamp': datetime.now()
    }
    clima_ref.document().set(data)


def read_data():
    docs = clima_ref.get()
    for data in docs:
        print(f'ID:{data.id} => DATA: {data.to_dict()}')

def update_data(id, t,h,p,l):
    data = {
        'temp': t,
        'hum': h,
        'pres': p,
        'led': l
    }
    clima_ref.document(id).update(data)

def delete_data(id):
    clima_ref.document(id).delete()

# ESTO YA ES PARA OTRA COLECCION (ACTUALES)
def update_actuales(t,h,p,l):
    data = {
        'temp': t,
        'hum': h,
        'pres': p,
        'led': l
    }
    actuales_ref.update(data)



temp = float(input("Ingresa temperatura: "))
pres = float(input("Ingresa presion: "))
hum = input("Ingresa humedad: ")

create_data(temp,pres,hum)
update_data(temp,pres,hum,True)