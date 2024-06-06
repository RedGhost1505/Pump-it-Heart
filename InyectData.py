import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import random

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()
# user_data_ref = db.collection('/users/375ygoLc9UaasBXnCWknLMvhQAt1/data')

user_data_ref = db.collection('users').document(
    'bxNlOgPt88gX9e3awXabEj4jzqA2').collection('data')

MET = 6.0 # MET value for barbell squat
weight = 70 # weight in kg
time = 10 # time in minutes


def kcal_burned(weight, time, MET):
    return round((MET * 3.5 * weight) / 200 * time, 2)

def create_data(type_exercise, avg_time, errors, performance_bar, performance_correct, performance_neck, reps, set_time):
    data = {
        'type_exercise': type_exercise,
        'avg_time': avg_time,
        'errors': errors,
        'performance_bar': performance_bar,
        'performance_correct': performance_correct,
        'performance_neck': performance_neck,
        'reps': reps,
        'set_time': set_time,
        'timestamp': datetime.now(),
        'kcal': kcal_burned(weight, set_time/60, MET)
    }
    user_data_ref.document().set(data)


def read_data():
    docs = user_data_ref.get()
    for data in docs:
        print(f'ID:{data.id} => DATA: {data.to_dict()}')


for i in range(1):
    create_data(type_exercise='barbell squat',
                avg_time=20,
                errors=0,
                performance_bar=5,
                performance_correct=5,
                performance_neck=5,
                reps=35,
                set_time=5)

# read_data()
