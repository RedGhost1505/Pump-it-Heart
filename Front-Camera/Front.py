import cv2
import mediapipe as mp
import math
import numpy as np

# 20 y 19 puntos de manos

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose()

# Contador de las veces que la flag cambia de True a False o viceversa
contador_cambios = 0

# Estado anterior de la flag para comparar con el estado actual en cada iteración
estado_anterior_flag = None

# file_name = 'Reference 2.mp4'
# cap = cv2.VideoCapture(file_name)
cap = cv2.VideoCapture(1)
#captura
while cap.isOpened():
    success,esqueleto = cap.read()
    if success:
        height, width, _ = esqueleto.shape 
        results = pose.process(esqueleto)
        if results.pose_landmarks:
            
            mp_drawing.draw_landmarks(esqueleto,results.pose_landmarks,mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2), connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
            cv2.imshow('esqueleto', esqueleto)

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

            cv2.circle(esqueleto, (point7), 3, (255,0,0), cv2.FILLED)  
            cv2.circle(esqueleto, (point8), 3, (255,0,0), cv2.FILLED)  
            cv2.line(esqueleto, (point7), (point8), (0, 255, 0), 5)
            
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

            dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
            magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
            magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)
            cosine_angle = dot_product / (magnitude1 * magnitude2)
            angle1 = math.degrees(math.acos(cosine_angle))

            dot_product = vector3[0] * vector4[0] + vector3[1] * vector4[1]
            magnitude11 = math.sqrt(vector3[0] ** 2 + vector3[1] ** 2)
            magnitude22 = math.sqrt(vector4[0] ** 2 + vector4[1] ** 2)
            cosine_angle1 = dot_product / (magnitude11 * magnitude22)
            angle11 = math.degrees(math.acos(cosine_angle1))

            angle=abs (angle1-180)
            angle2=abs (angle11-180)

            # Contador de repetición
            Indicator = ""
            FLAG = False

            if 160 <= angle and angle2 <= 180:
                Indicator = "Up"
                FLAG = True
            elif 30 <= angle and angle2 <= 90:
                Indicator = "Down"
                FLAG = False
            
            # Comprobar si esta es la primera iteración
            if estado_anterior_flag is None:
                estado_anterior_flag = FLAG
            else:
                # Si el estado de la flag ha cambiado, incrementar el contador
                if FLAG != estado_anterior_flag:
                    estado_anterior_flag = FLAG
                    if estado_anterior_flag == True:
                        contador_cambios += 1
                        # estado_anterior_flag = FLAG


            print(angle, angle2, Indicator, contador_cambios)

            cv2.putText(canva, str(int(angle)), (x2 - 50, y2 - 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            contours = np.array([[x1, y1], [x2, y2], [x3, y3]])
            cv2.fillPoly(canva, pts=[contours], color=(242, 233,228 ))

            cv2.putText(canva, str(int(angle2)), (x22 - 50, y22 - 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            contours = np.array([[x11, y11], [x22, y22], [x33, y33]])
            cv2.fillPoly(canva, pts=[contours], color=(242, 233,228 ))

            cv2.putText(canva, f'Contador de repeticiones: {contador_cambios}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_texto, 1)
            cv2.putText(esqueleto, f'Contador de repeticiones: {contador_cambios}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color_texto, 2)
            cv2.putText(esqueleto, f'Linea Manos: {texto_estado}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_texto, 1)

            cv2.imshow('esqueleto', esqueleto)
            cv2.imshow('Canva', canva)

        if cv2.waitKey(1) & 0xFF == ord ('q'):
            break
            
cap.release()
cv2.destroyAllWindows() 