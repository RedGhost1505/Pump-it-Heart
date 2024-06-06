from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QMainWindow
from pantalla_principal import Ui_MainWindow  # Asume que tu archivo se llama Pump_it_Front.py
from Dos_Pantalla import Ui_SecondaryWindow
from Instructions import Instructions_UI
from Resume import Resume_UI
import mediapipe as mp
import math
import numpy as np
import cv2 
import sys
import pygame
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

contador_cambios_global = 0
contador_de_errores = 0
Avg_time = 20
Tot_time = 20
MET = 6.0 # MET value for barbell squat
weight = 70 # weight in kg
time = 10 # time in minutes
performance_bar = 15
performance_correct = 15
performance_neck = 15
kcal = 50


class camThread(QThread):
    openResume = pyqtSignal(bool)
    change_pixmap_signal = pyqtSignal(QImage)
    contador_cambios_signal = pyqtSignal(int)
    
    def __init__(self, previewName, camID, model_type):
        super().__init__()
        self.previewName = previewName
        self.camID = camID
        self.model_type = model_type
        self.running = True

    def run(self):
        print("Starting " + self.previewName)
        if self.model_type == 'front':
            self.camPreview_Front()
        elif self.model_type == 'lat':
            self.camPreview_Lat()

    def camPreview_Front(self):
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()
        # Estado anterior de la flag para comparar con el estado actual en cada iteración
        estado_anterior_flag = None

        # Variables para el botón
        time_inside_button = 0
        tiempo_minimo_dentro_botón = 1  # Tiempo mínimo en segundos que la mano debe estar dentro del botón para activar la acción
        sound_played = False
        pygame.mixer.init()
        sound = pygame.mixer.Sound('multi-pop-5.mp3')

        def calcular_angulo_entre_vectores(vector1, vector2):
            dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
            magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
            magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)
            
            # Asegurarse de que las magnitudes no sean cero
            if magnitude1 == 0 or magnitude2 == 0:
                return 0, 180  # Retornar un ángulo neutral o seguro
            
            # Asegurar que el coseno esté en el intervalo [-1, 1]
            cosine_angle = dot_product / (magnitude1 * magnitude2)
            cosine_angle = max(-1, min(1, cosine_angle))  # Clamping the value

            try:
                angle = math.degrees(math.acos(cosine_angle))
            except ValueError:
                angle = 0  # O manejar de alguna otra manera adecuada

            angle_180 = abs(angle - 180)
            return angle, angle_180

        # cv2.namedWindow(self.previewName)
        cap = cv2.VideoCapture(self.camID)
        fps = cap.get(cv2.CAP_PROP_FPS)
        while self.running and cap.isOpened():
            success,esqueleto = cap.read()
            if not success:
                print("Error: No se puede leer la imagen de la cámara.")
                self.running = False
                break
            
            if success:
                height, width, _ = esqueleto.shape 
                esqueleto = cv2.flip(esqueleto, 1)
                results = pose.process(esqueleto)
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(esqueleto,results.pose_landmarks,mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2), connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
                    # cv2.imshow('esqueleto', esqueleto)

                    x1= int(results.pose_landmarks.landmark[24].x * width)#righthip
                    y1= int(results.pose_landmarks.landmark[24].y * height)#righthip
                    # print(f'X1:{x1},Y1:{y1}')

                    x2= int(results.pose_landmarks.landmark[26].x * width)#rightknee
                    y2= int(results.pose_landmarks.landmark[26].y * height)#rightknee
                    # print(f'X2:{x2},Y2:{y2}')

                    x3= int(results.pose_landmarks.landmark[28].x * width)#rightankle 
                    y3= int(results.pose_landmarks.landmark[28].y * height)#rightankle
                    # print(f'X3:{x3},Y3:{y3}')


                    x11= int(results.pose_landmarks.landmark[23].x * width)#leftthip
                    y11= int(results.pose_landmarks.landmark[23].y * height)#leftthip
                    # print(f'X11:{x11},Y11:{y11}')

                    x22= int(results.pose_landmarks.landmark[25].x * width)#leftknee
                    y22= int(results.pose_landmarks.landmark[25].y * height)#leftknee
                    # print(f'X22:{x22},Y22:{y22}')

                    x33= int(results.pose_landmarks.landmark[27].x * width)#leftankle
                    y33= int(results.pose_landmarks.landmark[27].y * height)#leftankle
                    # print(f'X33:{x33},Y33:{y33}')

                    #Puntos para manos 20 y 19
                    x20 = int(results.pose_landmarks.landmark[20].x * width)
                    y20 = int(results.pose_landmarks.landmark[20].y * height)

                    x19 = int(results.pose_landmarks.landmark[19].x * width)
                    y19 = int(results.pose_landmarks.landmark[19].y * height)

                    #Pies
                    x32 = int(results.pose_landmarks.landmark[32].x * width)
                    y32 = int(results.pose_landmarks.landmark[32].y * height)

                    x30 = int(results.pose_landmarks.landmark[30].x * width)
                    y30 = int(results.pose_landmarks.landmark[30].y * height)

                    x31 = int(results.pose_landmarks.landmark[31].x * width)
                    y31 = int(results.pose_landmarks.landmark[31].y * height)

                    x29 = int(results.pose_landmarks.landmark[29].x * width)
                    y29 = int(results.pose_landmarks.landmark[29].y * height)

                    # Left foot index.
                    l_foot_index_x = int(results.pose_landmarks.landmark[31].x * width)
                    l_foot_index_y = int(results.pose_landmarks.landmark[31].y * height)

                    # Right foot index.
                    r_foot_index_x = int(results.pose_landmarks.landmark[32].x * width)
                    r_foot_index_y = int(results.pose_landmarks.landmark[32].y * height)

                    # Left shoulder.
                    l_shldr_x = int(results.pose_landmarks.landmark[11].x * width)
                    l_shldr_y = int(results.pose_landmarks.landmark[11].y * height)

                    # Right shoulder.
                    r_shldr_x = int(results.pose_landmarks.landmark[12].x * width)
                    r_shldr_y = int(results.pose_landmarks.landmark[12].y * height)
                    

                    point1 = (x1, y1)#righthip
                    point2 = (x2, y2)#righsknee
                    point3 = (x3, y3)#rightankle
                    point4 = (x11, y11)#leftthip
                    point5 = (x22, y22)#leftknee
                    point6 = (x33, y33)#leftankle
                    point7 = (x20, y20)#mano derecha
                    point8 = (x19, y19)#mano izquierda
                    point9 = (x32, y32)#pie derecho
                    point10 = (x30, y30)#pie derecho
                    point11 = (x31, y31)#pie izquierdo
                    point12 = (x29, y29)#pie izquierdo
                    point13 = (l_foot_index_x, l_foot_index_y)#left foot index
                    point14 = (r_foot_index_x, r_foot_index_y)#right foot index
                    point15 = (l_shldr_x, l_shldr_y)#left shoulder
                    point16 = (r_shldr_x, r_shldr_y)#right shoulder


                    canva = np.zeros (esqueleto.shape, np.uint8) #matriz ceros negra fondo

                    # Calcula la pendiente entre los puntos de las manos
                    pendiente = (y19 - y20) / (x19 - x20) if (x19 - x20) != 0 else 0

                    # Define un umbral para considerar la línea como nivelada. 
                    # Este valor es un ejemplo y puede necesitar ajustes.
                    umbral_pendiente = 0.1

                    # Determina si la línea está nivelada comparando la pendiente con el umbral
                    esta_nivelada = abs(pendiente) < umbral_pendiente

                    # Visualiza el resultado
                    texto_estado = 'Nivelada' if esta_nivelada else 'No Nivelada'
                    color_texto = (0, 255, 0) if esta_nivelada else (0, 0, 255)

                    # Dibujar punto entre mano izquierda y derecha
                    cv2.circle(canva, (point7), 3, (255,0,0), cv2.FILLED)  
                    cv2.circle(canva, (point8), 3, (255,0,0), cv2.FILLED)  
                    cv2.line(canva, (point7), (point8), (0, 255, 0), 5)    

                    if esta_nivelada:
                        # Dibujar línea entre las manos
                        cv2.line(esqueleto, (point7), (point8), (0, 255, 0), 5)
                    else:
                        # Dibujar línea entre las manos
                        cv2.line(esqueleto, (point7), (point8), (0, 0, 255), 5)
                    
                    #Pierna derecha
                    cv2.line(canva, (point1), (point2), (0, 255, 0), 20)
                    cv2.line(canva, (point2), (point3), (0, 255, 0), 20)
                    cv2.line(canva, (point1), (point3), (0, 255, 0), 20)

                    #Pierna izquierda
                    cv2.line(canva, (point4), (point5), (0, 255, 0), 20)
                    cv2.line(canva, (point5), (point6), (0, 255, 0), 20)
                    cv2.line(canva, (point4), (point6), (0, 255, 0), 20)
                    
                    cv2.circle(esqueleto, (x1, y1), 6, (255,0,0),cv2.FILLED)#righthip
                    cv2.putText(esqueleto, 'cadera derecha', (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 206, 208), 2)
                
                    cv2.circle(esqueleto, (x2, y2), 6, (255,0,0),cv2.FILLED)#rightshoulder
                    cv2.putText(esqueleto, 'rodilla derecha', (x2, y2), cv2.FONT_HERSHEY_PLAIN, 1, (0, 206, 208), 2)
                    
                    cv2.circle(esqueleto, (x3, y3), 6, (255,0,0),cv2.FILLED)#rightelbow
                    cv2.putText(esqueleto, 'talón derecho', (x3, y3), cv2.FONT_HERSHEY_PLAIN, 1, (0, 206, 208), 2)

                    cv2.circle(esqueleto, (x11, y11), 6, (255,0,0),cv2.FILLED)#righthip
                    cv2.putText(esqueleto, 'cadera izquierda', (x11, y11), cv2.FONT_HERSHEY_PLAIN, 1, (0, 206, 208), 2)

                    cv2.circle(esqueleto, (x22, y22), 6, (255,0,0),cv2.FILLED)#rightshoulder
                    cv2.putText(esqueleto, 'rodilla izquierda', (x22, y22), cv2.FONT_HERSHEY_PLAIN, 1, (0, 206, 208), 2)

                    cv2.circle(esqueleto, (x33, y33), 6, (255,0,0),cv2.FILLED)#rightelbow
                    cv2.putText(esqueleto, 'talón izquierdo', (x33, y33), cv2.FONT_HERSHEY_PLAIN, 1, (0, 206, 208), 2)

                    cv2.line(esqueleto, point1, point2, (0, 255, 0), 2) 
                    cv2.line(esqueleto, point2, point3, (0, 255, 0), 2) 
                    cv2.line(esqueleto, point4, point5, (0, 255, 0), 2) 
                    cv2.line(esqueleto, point5, point6, (0, 255, 0), 2) 

                    vector1 = (point2[0] - point1[0], point2[1] - point1[1])
                    vector2 = (point3[0] - point2[0], point3[1] - point2[1])
                    vector3 = (point5[0] - point4[0], point5[1] - point4[1])
                    vector4 = (point6[0] - point5[0], point6[1] - point5[1])

                    angle1, angle1_adjusted = calcular_angulo_entre_vectores(vector1, vector2)
                    angle2, angle2_adjusted = calcular_angulo_entre_vectores(vector3, vector4)

                    cv2.putText(canva, str(int(angle1_adjusted)), (x2 - 50, y2 - 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                    contours = np.array([[x1, y1], [x2, y2], [x3, y3]])
                    cv2.fillPoly(canva, pts=[contours], color=(242, 233,228 ))

                    cv2.putText(canva, str(int(angle2_adjusted)), (x22 - 50, y22 - 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                    contours = np.array([[x11, y11], [x22, y22], [x33, y33]])
                    cv2.fillPoly(canva, pts=[contours], color=(242, 233,228 ))

                    # cv2.putText(canva, f'Contador de repeticiones: {contador_cambios}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_texto, 1)
                    # cv2.putText(esqueleto, f'Contador de repeticiones: {contador_cambios}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color_texto, 2)
                    cv2.putText(esqueleto, f'Tu barra se encuentra: {texto_estado}', (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_texto, 2)

                    # Áreas de los botones
                    menu_button_area = (20, 20, 120, 60)  # x1, y1, x2, y2 (esquina superior izquierda y esquina inferior derecha)
                    reset_button_area = (width - 120, 20, width - 20, 60)  # x1, y1, x2, y2 (esquina superior izquierda y esquina inferior derecha)

                    # Función para verificar si dos puntos está dentro de un área definida
                    def point_inside_area(point, area):
                        x, y = point
                        x1, y1, x2, y2 = area
                        return x1 <= x <= x2 and y1 <= y <= y2


                    # Dibujar los botones en la pantalla
                    cv2.rectangle(esqueleto, menu_button_area[:2], menu_button_area[2:], (0, 255,0 ), 2)
                    cv2.rectangle(esqueleto, reset_button_area[:2], reset_button_area[2:], (0, 255, 0), 2)

                    # Dibujar los botones en la pantalla
                    cv2.rectangle(esqueleto, menu_button_area[:2], menu_button_area[2:], (0, 0, 0), 2)
                    cv2.rectangle(esqueleto, reset_button_area[:2], reset_button_area[2:], (0, 0, 0), 2)

                    # Cargar las imágenes para representar las acciones
                    menu_image = cv2.imread('VOLVER.jpg')
                    reset_image = cv2.imread('REINICIAR.jpg')

                    # Redimensionar las imágenes si es necesario para que se ajusten al tamaño de los botones
                    menu_image = cv2.resize(menu_image, (menu_button_area[2] - menu_button_area[0], menu_button_area[3] - menu_button_area[1]))
                    reset_image = cv2.resize(reset_image, (reset_button_area[2] - reset_button_area[0], reset_button_area[3] - reset_button_area[1]))

                    # Dibujar las imágenes en los botones correspondientes
                    esqueleto[menu_button_area[1]:menu_button_area[3], menu_button_area[0]:menu_button_area[2]] = menu_image
                    esqueleto[reset_button_area[1]:reset_button_area[3], reset_button_area[0]:reset_button_area[2]] = reset_image

                    white_color = (255, 255, 255)

                    # Verificar interacciones con los botones
                    if point_inside_area(point7, menu_button_area) or point_inside_area(point8, menu_button_area):
                        # Incrementar el temporizador si la mano está dentro del área del botón
                        # Cambiar el color del contorno del botón al color blanco
                        cv2.rectangle(esqueleto, menu_button_area[:2], menu_button_area[2:], white_color, 2)
                        time_inside_button += 1 / fps  # fps es la velocidad de cuadros por segundo del video
                        if time_inside_button >= tiempo_minimo_dentro_botón:
                            # print("Volver al menú principal")
                            self.openResume.emit(True)

                            if not sound_played:
                                sound.play()
                                sound_played = True
                        

                    elif point_inside_area(point8, reset_button_area) or point_inside_area(point7, reset_button_area):
                        # Incrementar el temporizador si la mano está dentro del área del botón
                        # Cambiar el color del contorno del botón al color blanco
                        cv2.rectangle(esqueleto, reset_button_area[:2], reset_button_area[2:], white_color, 2)
                        time_inside_button += 1 / fps
                        if time_inside_button >= tiempo_minimo_dentro_botón:
                            # print("Reiniciar contador de repeticiones")
                            global contador_cambios_global
                            contador_cambios_global = 0
                            self.contador_cambios_signal.emit(contador_cambios_global)
                            if not sound_played:
                                sound.play()
                                sound_played = True
            
                    else:
                        # Restablecer el temporizador si la mano no está dentro del área del botón
                        time_inside_button = 0
                        sound_played = False

                    # Validar posición de las piernas para indicar si están bien colocadas o no 

                    if point5 and point2 and point13 and point14 and point15 and point16:
                        if point15[0]+40 < point5[0] < point15[0]+20 and point16[0]-20 < point2[0] < point16[0]-40:
                            print('Piernas bien colocadas')
                        else:
                            cv2.rectangle(esqueleto, (point15[0]-4, point5[1]+2), (point15[0]+40, point5[1]-20), (255,0,0),cv2.FILLED)#leftknee
                            cv2.rectangle(esqueleto, (point16[0]-4, point2[1]+2), (point16[0]-40, point2[1]-20), (255,0,0),cv2.FILLED)#rightknee

                else:
                    cv2.putText(esqueleto, 'No Landmarks Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                rgb_image = cv2.cvtColor(esqueleto, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                # p = convert_to_Qt_format.scaled(640, 480, aspectMode = QtCore.Qt.AspectRatioMode.KeepAspectRatio)
                self.change_pixmap_signal.emit(convert_to_Qt_format)

    def camPreview_Lat(self):

        estado_anterior_flag = None
        iteration = False

        # Flag
        Indicator = ""
        FLAG = False
        full_cycle_completed = False  # Indicador de ciclo completo
        coordenatesFootStart = (0, 0)
        coordenatesFootEnd = (0, 0)

        # Calculate distance
        def findDistance(x1, y1, x2, y2):
            dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            return dist


        # Calculate angle.
        def findAngle(x1, y1, x2, y2):
            # Calcular la diferencia en coordenadas
            dx = x2 - x1
            dy = y2 - y1
            # Calcular la distancia entre los puntos
            distance = math.sqrt(dx ** 2 + dy ** 2)

            # Verificar si la distancia o y1 es cero
            if distance == 0 or y1 == 0:
                return 0  # Ángulo indefinido o cero
            
            # Calcular el coseno del ángulo usando la fórmula segura
            cos_theta = ((dy * -y1) / (distance * y1))
            # Asegurar que el coseno esté en el rango permitido para acos
            cos_theta = max(-1, min(1, cos_theta))
            
            # Calcular el ángulo en radianes y luego convertir a grados
            theta = math.acos(cos_theta)
            degree = theta * (180 / math.pi)
            
            return degree
        
        def calcular_angulo_entre_vectores(vector1, vector2):
            dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
            magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
            magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)
            
            # Asegurarse de que las magnitudes no sean cero
            if magnitude1 == 0 or magnitude2 == 0:
                return 0, 180  # Retornar un ángulo neutral o seguro
            
            # Asegurar que el coseno esté en el intervalo [-1, 1]
            cosine_angle = dot_product / (magnitude1 * magnitude2)
            cosine_angle = max(-1, min(1, cosine_angle))  # Clamping the value

            try:
                angle = math.degrees(math.acos(cosine_angle))
            except ValueError:
                angle = 0  # O manejar de alguna otra manera adecuada

            angle_180 = abs(angle - 180)
            return angle, angle_180
        
        """
        Function to send alert. Use this function to send alert when bad posture detected.
        Feel free to get creative and customize as per your convenience.
        """

        def sendWarning(x):
            pass

        # =============================CONSTANTS and INITIALIZATIONS=====================================#
        # Initilize frame counters.
        good_frames = 0
        bad_frames = 0

        # Font type.
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Colors.
        blue = (255, 127, 0)
        red = (50, 50, 255)
        green = (127, 255, 0)
        dark_blue = (127, 20, 0)
        light_green = (127, 233, 100)
        yellow = (0, 255, 255)
        pink = (255, 0, 255)

        # Initialize mediapipe pose class.
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()
        # ===============================================================================================#

        # cv2.namedWindow(self.previewName)
        cam = cv2.VideoCapture(self.camID)

        # Meta.
        fps = int(cam.get(cv2.CAP_PROP_FPS))
        width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (width, height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Video writer.
        video_output = cv2.VideoWriter('output.mp4', fourcc, fps, frame_size)

        while self.running and cam.isOpened():
            # Capture frames.
            success, image = cam.read()
            if not success:
                self.running = False
                break
            # Get fps.
            fps = cam.get(cv2.CAP_PROP_FPS)
            # Get height and width.
            h, w = image.shape[:2]
            #Print the height and width of the frame
            # print(f'Height: {h}, Width: {w}')

            # Convert the BGR image to RGB.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the image.
            keypoints = pose.process(image)

            # Convert the image back to BGR.
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if keypoints.pose_landmarks:

                # Use lm and lmPose as representative of the following methods.
                lm = keypoints.pose_landmarks
                lmPose = mp_pose.PoseLandmark

                # Acquire the landmark coordinates.
                # Once aligned properly, left or right should not be a concern.      
                # Left shoulder.
                l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
                l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
                # Right shoulder
                r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
                r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
                # Left ear.
                l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
                l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)
                # Left hip.
                l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
                l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

                # Left knee.
                l_knee_x = int(lm.landmark[lmPose.LEFT_KNEE].x * w)
                l_knee_y = int(lm.landmark[lmPose.LEFT_KNEE].y * h)
                # Left ankle.
                l_ankle_x = int(lm.landmark[lmPose.LEFT_ANKLE].x * w)
                l_ankle_y = int(lm.landmark[lmPose.LEFT_ANKLE].y * h)
                # Left heel.
                l_heel_x = int(lm.landmark[lmPose.LEFT_HEEL].x * w)
                l_heel_y = int(lm.landmark[lmPose.LEFT_HEEL].y * h)
                # Left foot index.
                l_foot_index_x = int(lm.landmark[lmPose.LEFT_FOOT_INDEX].x * w)
                l_foot_index_y = int(lm.landmark[lmPose.LEFT_FOOT_INDEX].y * h)

                heel_height = abs(l_heel_y - l_foot_index_y)
                # Threshold for heel lift detection.
                threshold = 10  # Adjust this value as needed.

                # Check for heel lift and display alert.
                if heel_height > threshold:
                    cv2.putText(image, 'Talon: False', (l_heel_x + 10, l_heel_y), font, 0.9, red, 2)
                    cv2.line(image, (l_ankle_x, l_ankle_y), (l_heel_x, l_heel_y), red, 4)
                else:
                    cv2.putText(image, 'Talon: True', (l_heel_x + 10, l_heel_y), font, 0.9, green, 2)
                    cv2.line(image, (l_ankle_x, l_ankle_y), (l_heel_x, l_heel_y), green, 4)

                # Calculate distance between left shoulder and right shoulder points.
                offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)

                # Assist to align the camera to point at the side view of the person.
                # Offset threshold 30 is based on results obtained from analysis over 100 samples.
                if offset < 100:
                    cv2.putText(image, str(int(offset)) + ' Aligned', (w - 150, 30), font, 0.9, green, 2)
                else:
                    cv2.putText(image, str(int(offset)) + ' Not Aligned', (w - 150, 30), font, 0.9, red, 2)

                # Calculate angles.
                neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
                torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)
                Vector1 = [l_knee_x - l_hip_x, l_knee_y - l_hip_y]
                Vector2 = [l_ankle_x - l_knee_x, l_ankle_y - l_knee_y]
                leg_Angle = calcular_angulo_entre_vectores(Vector1, Vector2)[1]

                # Draw landmarks.
                cv2.circle(image, (l_shldr_x, l_shldr_y), 7, yellow, -1)
                cv2.circle(image, (l_ear_x, l_ear_y), 7, yellow, -1)
                cv2.circle(image, (l_hip_x, l_hip_y), 7, yellow, -1)
                cv2.circle(image, (l_knee_x, l_knee_y), 7, yellow, -1)
                cv2.circle(image, (l_ankle_x, l_ankle_y), 7, yellow, -1)
                cv2.circle(image, (l_heel_x, l_heel_y), 7, yellow, -1)
                cv2.circle(image, (l_foot_index_x, l_foot_index_y), 7, yellow, -1)

                # Let's take y - coordinate of P3 100px above x1,  for display elegance.
                # Although we are taking y = 0 while calculating angle between P1,P2,P3.
                cv2.circle(image, (l_shldr_x, l_shldr_y - 100), 7, yellow, -1)
                cv2.circle(image, (r_shldr_x, r_shldr_y), 7, pink, -1)
                cv2.circle(image, (l_hip_x, l_hip_y), 7, yellow, -1)

                # Similarly, here we are taking y - coordinate 100px above x1. Note that
                # you can take any value for y, not necessarily 100 or 200 pixels.
                cv2.circle(image, (l_hip_x, l_hip_y - 100), 7, yellow, -1)

                # Put text, Posture and angle inclination.
                # Text string for display.
                angle_text_string = 'Neck : ' + str(int(neck_inclination)) + '  Torso : ' + str(int(torso_inclination))

                # Determine whether good posture or bad posture.
                # The threshold angles have been set based on intuition.
                if neck_inclination < 55 and torso_inclination < 65:
                    bad_frames = 0
                    good_frames += 1
                    
                    cv2.putText(image, angle_text_string, (10, 30), font, 0.9, light_green, 2)
                    cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, light_green, 2)
                    cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, light_green, 2)
                    cv2.putText(image, str(int(leg_Angle)), (l_knee_x + 10, l_knee_y), font, 0.9, light_green, 2)

                    # Join landmarks.
                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), green, 4)
                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), green, 4)
                    cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), green, 4)
                    cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), green, 4)
                    cv2.line(image, (l_knee_x, l_knee_y), (l_ankle_x, l_ankle_y), green, 4)
                    cv2.line(image, (l_ankle_x, l_ankle_y), (l_heel_x, l_heel_y), green, 4)
                    cv2.line(image, (l_ankle_x, l_ankle_y), (l_foot_index_x, l_foot_index_y), green, 4)
                    cv2.line(image, (l_hip_x, l_hip_y), (l_knee_x, l_knee_y), green, 4)
                    cv2.line(image,(l_foot_index_x, l_foot_index_y), (l_heel_x, l_heel_y), green, 4)
                    cv2.line(image,(l_hip_x, l_hip_y), (l_ankle_x, l_ankle_y), green, 4)

                else:
                    good_frames = 0
                    bad_frames += 1

                    cv2.putText(image, angle_text_string, (10, 30), font, 0.9, red, 2)
                    cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, red, 2)
                    cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, red, 2)
                    cv2.putText(image, str(int(leg_Angle)), (l_knee_x + 10, l_knee_y), font, 0.9, red, 2)

                    # Join landmarks.
                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), red, 4)
                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), red, 4)
                    cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), red, 4)
                    cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), red, 4)
                    cv2.line(image, (l_knee_x, l_knee_y), (l_ankle_x, l_ankle_y), red, 4)
                    cv2.line(image, (l_ankle_x, l_ankle_y), (l_heel_x, l_heel_y), red, 4)
                    cv2.line(image, (l_ankle_x, l_ankle_y), (l_foot_index_x, l_foot_index_y), red, 4)
                    cv2.line(image, (l_hip_x, l_hip_y), (l_knee_x, l_knee_y), red, 4)
                    cv2.line(image,(l_foot_index_x, l_foot_index_y), (l_heel_x, l_heel_y), red, 4)
                    cv2.line(image,(l_hip_x, l_hip_y), (l_ankle_x, l_ankle_y), red, 4)

                # Calculate the time of remaining in a particular posture.
                good_time = (1 / fps) * good_frames
                bad_time =  (1 / fps) * bad_frames

                # Pose time.
                if good_time > 0:
                    time_string_good = 'Good Posture Time : ' + str(round(good_time, 1)) + 's'
                    cv2.putText(image, time_string_good, (10, h - 20), font, 0.9, green, 2)
                else:
                    time_string_bad = 'Bad Posture Time : ' + str(round(bad_time, 1)) + 's'
                    cv2.putText(image, time_string_bad, (10, h - 20), font, 0.9, red, 2)

                # If you stay in bad posture for more than 3 minutes (180s) send an alert.
                if bad_time > 180:
                    sendWarning()
                # Write frames.
                video_output.write(image)

                if 160 <= leg_Angle <= 180:
                    if Indicator == "Down":  # Cambio de "Down" a "Up"
                        full_cycle_completed = True
                    Indicator = "Up"
                    FLAG = False
                    # print("Up")

                elif 30 <= leg_Angle <= 90:
                    if Indicator == "Up" and full_cycle_completed:  # Solo contar si antes estaba "Up" y se completó un ciclo
                        global contador_cambios_global
                        contador_cambios_global += 1
                        print("Entró al contador de cambios")
                        self.contador_cambios_signal.emit(contador_cambios_global)
                        full_cycle_completed = False  # Restablecer el indicador de ciclo completo
                    Indicator = "Down"
                    FLAG = True
                    # print("Down")

                # Check if the flag changes from True to False or vice versa
                if l_foot_index_x and l_foot_index_y:
                    if estado_anterior_flag is None:
                        estado_anterior_flag = FLAG
                    elif estado_anterior_flag != FLAG:
                        estado_anterior_flag = FLAG
                        


            else:
                cv2.putText(image, 'No Landmarks Detected', (10, 30), font, 0.9, blue, 2)

            # Display.
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytesPerLine = ch * w
            convertToQtFormat = QImage(rgb_image.data, w, h, bytesPerLine, QImage.Format.Format_RGB888)
            # p = convertToQtFormat.scaled(640, 480, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            self.change_pixmap_signal.emit(convertToQtFormat)
    
    def stop(self):
        self.running = False
        self.quit()
        self.wait()


#-------------------------------------------------------------
#PyQt6

class SecondaryWindow(QMainWindow):
    def __init__(self, parent=None):
        super(SecondaryWindow, self).__init__(parent)
        self.ui = Ui_SecondaryWindow()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.openMainWindow)
        print ("Iniciando UI")


        # Contador de las veces que la flag cambia de True a False o viceversa
        self.contador_cambios = 0
        # Estado anterior de la flag para comparar con el estado actual en cada iteración
        self.estado_anterior_flag = None
        

        self.thread1 = camThread("Front", 0, 'front')
        self.thread1.change_pixmap_signal.connect(self.update_image1)
        self.thread1.contador_cambios_signal.connect(self.update_counter)
        self.thread1.openResume.connect(self.openResume)
        self.thread1.start()

        self.thread2 = camThread("Lateral", 1, 'lat')
        self.thread2.change_pixmap_signal.connect(self.update_image2)
        self.thread2.contador_cambios_signal.connect(self.update_counter)
        self.thread2.start()



    def update_image1(self, image):
        if image:
            pixmap = QPixmap.fromImage(image)
            if not pixmap.isNull():
                # print("Updating image 1")
                self.ui.imageLabel.setPixmap(pixmap)
            else:
                print("Error: Pixmap is null.")
        else:
            print("Error: Received image is null.")

    def update_image2(self, image):
        if image:
            pixmap = QPixmap.fromImage(image)
            if not pixmap.isNull():
                # print("Updating image 2")
                self.ui.imageLabel_2.setPixmap(pixmap)
            else:
                print("Error: Pixmap is null.")
        else:
            print("Error: Received image is null.")
    
    def update_counter(self, contador_cambios):
        self.ui.label_5.setText(f'{contador_cambios}')


    # Liberar la captura cuando la ventana se cierre
    def closeEvent(self, event):
        self.thread1.stop()
        self.thread2.stop()
        super().closeEvent(event)

    def openMainWindow(self):
        self.thread1.stop()
        self.thread2.stop()

        self.parent().show()
        self.close()

    def openResume(self):
        self.thread1.stop()
        self.thread2.stop()
        self.resumeWindow = ResumeWindow(self)
        self.resumeWindow.show()
        self.close()
                                

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        global contador_cambios_global
        contador_cambios_global = 0
        global contador_de_errores
        contador_de_errores = 0
        global Avg_time
        Avg_time = 15
        global Tot_time
        Tot_time = 15
        self.ui.pushButton.clicked.connect(self.openSecondaryScreen)
        self.ui.pushButton_2.clicked.connect(self.openInstructable)

    def openSecondaryScreen(self):
        self.secondaryWindow = SecondaryWindow(self)
        self.secondaryWindow.show()
        self.hide()
    
    def openInstructable(self):
        self.instructableWindow = InstructableWindow(self)
        self.instructableWindow.show()
        self.hide()

class InstructableWindow(QMainWindow):
    def __init__(self, parent=None):
        super(InstructableWindow, self).__init__(parent)
        self.ui = Instructions_UI()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.openMainWindow)

    def openMainWindow(self):
        self.parent().show()
        self.close()

class ResumeWindow(QMainWindow):
    def __init__(self, parent=None):
        super(ResumeWindow, self).__init__(parent)
        self.ui = Resume_UI()
        self.ui.setupUi(self)
        self.ui.Home.clicked.connect(self.openMainWindow)
        global contador_cambios_global
        self.ui.Reps.setText(str(contador_cambios_global))
        global contador_de_errores
        self.ui.Errors.setText(str(contador_de_errores))
        global Tot_time
        self.ui.Settime.setText(str(Tot_time))
        # Configuracion Firebase
        cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred)
        # Configuración de BD (Firestore)
        db = firestore.client()
        self.user_data_ref = db.collection('users').document('bxNlOgPt88gX9e3awXabEj4jzqA2').collection('data')

        global Avg_time
        global performance_bar
        global performance_correct
        global performance_neck
        global MET
        global weight
        global kcal


        
        # Inyección de datos en la BD
        data = {
            'type_exercise': 'barbell squat',
            'avg_time': Avg_time,
            'errors': contador_de_errores,
            'performance_bar': performance_bar,
            'performance_correct': performance_correct,
            'performance_neck': performance_neck,
            'reps': contador_cambios_global,
            'set_time': Tot_time,
            'timestamp': datetime.now(),
            'kcal': kcal,
        }
        self.user_data_ref.document().set(data)

    def openMainWindow(self):
        self.openMainWindow = MainWindow(self)
        self.openMainWindow.show()
        self.close()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec())
