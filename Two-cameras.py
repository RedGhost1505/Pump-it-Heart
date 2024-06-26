import cv2
import threading
import time
import math
import mediapipe as mp
import numpy as np

stop_event = threading.Event()
#------------------------------------------------------------

class camThread(threading.Thread):
    def __init__(self, previewName, camID, model_type):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID
        self.model_type = model_type
 
    def run(self):
        print("Starting " + self.previewName)
        if self.model_type == 'front':
            self.camPreview_Front()
        elif self.model_type == 'lat':
            self.camPreview_Lat()

#------------------------------------------------------------

    def camPreview_Front(self):
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()
        # Contador de las veces que la flag cambia de True a False o viceversa
        contador_cambios = 0
        # Estado anterior de la flag para comparar con el estado actual en cada iteración
        estado_anterior_flag = None

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

        cv2.namedWindow(self.previewName)
        cap = cv2.VideoCapture(self.camID)
        while not stop_event.is_set() and cap.isOpened():
            success,esqueleto = cap.read()
            if not success:
                stop_event.set()
                break
            
            if success:
                height, width, _ = esqueleto.shape 
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

                    angle1, angle1_adjusted = calcular_angulo_entre_vectores(vector1, vector2)
                    angle2, angle2_adjusted = calcular_angulo_entre_vectores(vector3, vector4)

                    # Los ángulos 'angle1_adjusted' y 'angle2_adjusted' ya están ajustados a 180 grados si es necesario
                    # Puedes utilizar 'angle1' y 'angle2' directamente si prefieres los ángulos originales sin ajustar

                    # Contador de repetición
                    Indicator = ""
                    FLAG = False

                    # Utiliza estos ángulos para cualquier procesamiento posterior, como evaluación de condiciones
                    if 160 <= angle1_adjusted and angle2_adjusted <= 180:
                        Indicator = "Up"
                        FLAG = True
                    elif 30 <= angle1_adjusted and angle2_adjusted <= 90:
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


                    # print(angle1_adjusted, angle2_adjusted, Indicator, contador_cambios)

                    cv2.putText(canva, str(int(angle1_adjusted)), (x2 - 50, y2 - 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                    contours = np.array([[x1, y1], [x2, y2], [x3, y3]])
                    cv2.fillPoly(canva, pts=[contours], color=(242, 233,228 ))

                    cv2.putText(canva, str(int(angle2_adjusted)), (x22 - 50, y22 - 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                    contours = np.array([[x11, y11], [x22, y22], [x33, y33]])
                    cv2.fillPoly(canva, pts=[contours], color=(242, 233,228 ))

                    cv2.putText(canva, f'Contador de repeticiones: {contador_cambios}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_texto, 1)
                    cv2.putText(esqueleto, f'Contador de repeticiones: {contador_cambios}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color_texto, 2)
                    cv2.putText(esqueleto, f'Linea Manos: {texto_estado}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_texto, 1)

                else:
                    cv2.putText(esqueleto, 'No Landmarks Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                #Display
                # cv2.imshow('Canva', canva)
                cv2.imshow(self.previewName, esqueleto)
                if cv2.waitKey(1) & 0xFF == ord('o'):
                    contador_cambios = 0
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()
                
        cap.release()
        cv2.destroyAllWindows() 


    def camPreview_Lat(self):

        # Calculate distance
        def findDistance(x1, y1, x2, y2):
            dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            return dist


        # Calculate angle.
        def findAngle(x1, y1, x2, y2):
            theta = math.acos((y2 - y1) * (-y1) / (math.sqrt(
                (x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
            degree = int(180 / math.pi) * theta
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

        cv2.namedWindow(self.previewName)
        cam = cv2.VideoCapture(self.camID)

        # Meta.
        fps = int(cam.get(cv2.CAP_PROP_FPS))
        width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (width, height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Video writer.
        video_output = cv2.VideoWriter('output.mp4', fourcc, fps, frame_size)

        while not stop_event.is_set() and cam.isOpened():
            # Capture frames.
            success, image = cam.read()
            if not success:
                stop_event.set()
                break
            # Get fps.
            fps = cam.get(cv2.CAP_PROP_FPS)
            # Get height and width.
            h, w = image.shape[:2]
            #Print the height and width of the frame
            print(f'Height: {h}, Width: {w}')

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

            else:
                cv2.putText(image, 'No Landmarks Detected', (10, 30), font, 0.9, blue, 2)

            # Display.
            cv2.imshow(self.previewName, image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                stop_event.set()

        cam.release()
        cv2.destroyAllWindows()

#------------------------------------------------------------


# Create two threads as follows
thread1 = camThread("Camera 1", 0, 'front')
thread2 = camThread("Camera 2", 1, 'lat')
thread1.start()
thread2.start()