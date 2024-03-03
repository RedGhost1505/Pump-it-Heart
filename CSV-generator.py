import cv2
import mediapipe as mp 
import matplotlib.pyplot as plt
import pandas as pd


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

#Dataframe
columns = ['Landmark', 'X','Y','Z']
df = pd.DataFrame(columns=columns)

def update_dataframe(landmark, x, y, z):
    global df
    new_row = pd.DataFrame({'Landmark': [landmark], 'X': [x], 'Y': [y], 'Z': [z]})
    df = pd.concat([df, new_row], ignore_index=True)


def main():
    file_name = 'Reference 2.mp4'
    cap = cv2.VideoCapture(file_name)
    while cap.isOpened(): 
        success, img = cap.read()
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(frame_rgb)
        mp_drawing.draw_landmarks(img, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2,circle_radius=2),connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
        
        # Verificar si se detectaron poses
        if pose_results.pose_landmarks:
            # Acceder a las coordenadas de los puntos de referencia
            landmarks = pose_results.pose_landmarks.landmark

            # Ejemplo: Imprimir las coordenadas del punto del hombro izquierdo (index 11)
            shoulder_left_x = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
            shoulder_left_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
            shoulder_left_z = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z  # En caso de que esté disponible la coordenada z

            ankle_left_x = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x
            ankle_left_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
            ankle_left_z = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z  

            ear_left_x = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x
            ear_left_y = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y
            ear_left_z = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].z 

            elbow_left_x = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x
            elbow_left_y = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y
            elbow_left_z = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z 
            
            eye_left_x = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].x
            eye_left_y = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].y
            eye_left_z = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].z 
            
            eye_inner_left_x = landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER.value].x
            eye_inner_left_y = landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER.value].y
            eye_inner_left_z = landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER.value].z 
            
            eye_outer_left_x = landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER.value].x
            eye_outer_left_y = landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER.value].y
            eye_outer_left_z = landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER.value].z 
            
            foot_index_left_x = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x
            foot_index_left_y = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y
            foot_index_left_z = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].z 
            
            heel_left_x = landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x
            heel_left_y = landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y
            heel_left_z = landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].z 
            
            hip_left_x = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x
            hip_left_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
            hip_left_z = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z 

            index_left_x = landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x
            index_left_y = landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y
            index_left_z = landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].z 

            knee_left_x = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x
            knee_left_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
            knee_left_z = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z 

            # mouth_left_x = landmarks[mp_pose.PoseLandmark.LEFT_MOUTH.value].x
            # mouth_left_y = landmarks[mp_pose.PoseLandmark.LEFT_MOUTH.value].y
            # mouth_left_z = landmarks[mp_pose.PoseLandmark.LEFT_MOUTH.value].z 

            pinky_left_x = landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].x
            pinky_left_y = landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].y
            pinky_left_z = landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].z 

            thumb_left_x = landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].x
            thumb_left_y = landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].y
            thumb_left_z = landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].z 

            wrist_left_x = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x
            wrist_left_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
            wrist_left_z = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z 
            
            nose_x = landmarks[mp_pose.PoseLandmark.NOSE.value].x
            nose_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y
            nose_z = landmarks[mp_pose.PoseLandmark.NOSE.value].z 

            ankle_right_x = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x
            ankle_right_y = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
            ankle_right_z = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z
            
            ear_right_x = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x
            ear_right_y = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y
            ear_right_z = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].z 

            elbow_right_x = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x
            elbow_right_y = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y
            elbow_right_z = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z 

            eye_right_x = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].x
            eye_right_y = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].y
            eye_right_z = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].z 

            eye_inner_right_x = landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].x
            eye_inner_right_y = landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].y
            eye_inner_right_z = landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].z 

            eye_outer_right_x = landmarks[mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value].x
            eye_outer_right_y = landmarks[mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value].y
            eye_outer_right_z = landmarks[mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value].z 

            foot_index_right_x = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x
            foot_index_right_y = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y
            foot_index_right_z = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].z 

            heel_right_x = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x
            heel_right_y = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y
            heel_right_z = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].z

            hip_right_x = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x
            hip_right_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
            hip_right_z = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z

            index_right_x = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x
            index_right_y = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y
            index_right_z = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].z

            knee_right_x = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x
            knee_right_y = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y
            knee_right_z = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z

            # mouth_right_x = landmarks[mp_pose.PoseLandmark.RIGHT_MOUTH.value].x
            # mouth_right_y = landmarks[mp_pose.PoseLandmark.RIGHT_MOUTH.value].y
            # mouth_right_z = landmarks[mp_pose.PoseLandmark.RIGHT_MOUTH.value].z

            pinky_right_x = landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].x
            pinky_right_y = landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].y
            pinky_right_z = landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].z
            
            shoulder_right_x = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
            shoulder_right_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
            shoulder_right_z = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z

            thumb_right_x = landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].x
            thumb_right_y = landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].y
            thumb_right_z = landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].z

            wrist_right_x = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x
            wrist_right_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
            wrist_right_z = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z

            for landmark in [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ANKLE,
                             mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.LEFT_ELBOW,
                             mp_pose.PoseLandmark.LEFT_EYE, mp_pose.PoseLandmark.LEFT_EYE_INNER,
                             mp_pose.PoseLandmark.LEFT_EYE_OUTER, mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
                             mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.LEFT_HIP,
                             mp_pose.PoseLandmark.LEFT_INDEX, mp_pose.PoseLandmark.LEFT_KNEE,
                             mp_pose.PoseLandmark.LEFT_PINKY, mp_pose.PoseLandmark.LEFT_THUMB,
                             mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.NOSE,
                             mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_EAR,
                             mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_EYE,
                             mp_pose.PoseLandmark.RIGHT_EYE_INNER, mp_pose.PoseLandmark.RIGHT_EYE_OUTER,
                             mp_pose.PoseLandmark.RIGHT_FOOT_INDEX, mp_pose.PoseLandmark.RIGHT_HEEL,
                             mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_INDEX,
                             mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_PINKY,
                             mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_THUMB,
                             mp_pose.PoseLandmark.RIGHT_WRIST]:

                x = landmarks[landmark.value].x
                y = landmarks[landmark.value].y
                z = landmarks[landmark.value].z if landmarks[landmark.value].HasField('z') else None

                update_dataframe(str(landmark), x, y, z)


        #Plano Izquierdo
        print(f"Coordenadas del hombro izquierdo - X: {shoulder_left_x}, Y: {shoulder_left_y}, Z: {shoulder_left_z}")
        print(f"Coordenadas del tobillo izquierdo - X: {ankle_left_x}, Y: {ankle_left_y}, Z: {ankle_left_z}")
        print(f"Coordenadas de oreja izquierda - X: {ear_left_x}, Y: {ear_left_y}, Z: {ear_left_z}")
        print(f"Coordenadas del codo izquierdo - X: {elbow_left_x}, Y: {elbow_left_y}, Z: {elbow_left_z}")
        print(f"Coordenadas del ojo izquierdo - X: {eye_left_x}, Y: {eye_left_y}, Z: {eye_left_z}")
        print(f"Coordenadas del ojo izquierdo (inner) - X: {eye_inner_left_x}, Y: {eye_inner_left_y}, Z: {eye_inner_left_z}")
        print(f"Coordenadas del ojo izquierdo (outer) - X: {eye_outer_left_x}, Y: {eye_outer_left_y}, Z: {eye_outer_left_z}")
        print(f"Coordenadas del pie izquierdo - X: {foot_index_left_x}, Y: {foot_index_left_y}, Z: {foot_index_left_z}")
        print(f"Coordenadas del tacon izquierdo - X: {heel_left_x}, Y: {heel_left_y}, Z: {heel_left_z}")
        print(f"Coordenadas de cadera izquierda - X: {hip_left_x}, Y: {hip_left_y}, Z: {hip_left_z}")
        print(f"Coordenadas de index izquierdo - X: {index_left_x}, Y: {index_left_y}, Z: {index_left_z}")
        print(f"Coordenadas de rodilla izquierda - X: {knee_left_x}, Y: {knee_left_y}, Z: {knee_left_z}")
        print(f"Coordenadas de meñique izquierdo - X: {pinky_left_x}, Y: {pinky_left_y}, Z: {pinky_left_z}")
        print(f"Coordenadas de pulgar izquierdo - X: {thumb_left_x}, Y: {thumb_left_y}, Z: {thumb_left_z}")
        print(f"Coordenadas del muñeca izquierdo - X: {wrist_left_x}, Y: {wrist_left_y}, Z: {wrist_left_z}")
        print(f"Coordenadas de nariz - X: {nose_x}, Y: {nose_y}, Z: {nose_z}")

        #Plano derecho
        print(f"Coordenadas del hombro derecho - X: {shoulder_right_x}, Y: {shoulder_right_y}, Z: {shoulder_right_z}")
        print(f"Coordenadas del tobillo derecho - X: {ankle_right_x}, Y: {ankle_right_y}, Z: {ankle_right_z}")
        print(f"Coordenadas de oreja derecha - X: {ear_right_x}, Y: {ear_right_y}, Z: {ear_right_z}")
        print(f"Coordenadas del codo derecho - X: {elbow_right_x}, Y: {elbow_right_y}, Z: {elbow_right_z}")
        print(f"Coordenadas del ojo derecho - X: {eye_right_x}, Y: {eye_right_y}, Z: {eye_right_z}")
        print(f"Coordenadas del ojo derecho (inner) - X: {eye_inner_right_x}, Y: {eye_inner_right_y}, Z: {eye_inner_right_z}")
        print(f"Coordenadas del ojo derecho (outer) - X: {eye_outer_right_x}, Y: {eye_outer_right_y}, Z: {eye_outer_right_z}")
        print(f"Coordenadas del pie derecho - X: {foot_index_right_x}, Y: {foot_index_right_y}, Z: {foot_index_right_z}")
        print(f"Coordenadas del tacon derecho - X: {heel_right_x}, Y: {heel_right_y}, Z: {heel_right_z}")
        print(f"Coordenadas de cadera derecha - X: {hip_right_x }, Y: {hip_right_y}, Z: {hip_right_z}")
        print(f"Coordenadas de index derecho - X: {index_right_x}, Y: {index_right_y}, Z: {index_right_z}")
        print(f"Coordenadas de rodilla derecha - X: {knee_right_x}, Y: {knee_right_y}, Z: {knee_right_z}")
        print(f"Coordenadas de meñique derecho - X: {pinky_right_x}, Y: {pinky_right_y}, Z: {pinky_right_z}")
        print(f"Coordenadas de pulgar derecho - X: {thumb_right_x}, Y: {thumb_right_y}, Z: {thumb_right_z}")
        print(f"Coordenadas del muñeca derecho - X: {wrist_right_x}, Y: {wrist_right_y}, Z: {wrist_right_z}")

        #Creación del CSV

        cv2.imshow("Image",img)
        key = cv2.waitKey(1) 
        if key == 27:
            break
    
    df.to_csv('Lets_Train.csv', index=False)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()