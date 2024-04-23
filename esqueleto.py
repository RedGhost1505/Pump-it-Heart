import cv2
import mediapipe as mp
import math
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose() #cargar el modelo MediaPipe Pose

cap = cv2.VideoCapture(0)
#captura
while cap.isOpened():
    success,esqueleto = cap.read()
    if success:
        height, width, _ = esqueleto.shape 
        results = pose.process(esqueleto)
        if results.pose_landmarks:
            
            mp_drawing.draw_landmarks(esqueleto,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,)
            cv2.imshow('esqueleto', esqueleto)
            

        if cv2.waitKey(1) & 0xFF == ord ('q'):
            break
cap.release()
cv2.destroyAllWindows()