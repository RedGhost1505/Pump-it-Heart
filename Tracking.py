import cv2
import mediapipe as mp 
import matplotlib.pyplot as plt



#        \:.             .:/
#         \``._________.''/ 
#          \             / 
#  .--.--, / .':.   .':. \
# /__:  /  | '::' . '::' |     Welcome to the pump-it app tracking test 
#    / /   |`.   ._.   .'|
#   / /    |.'         '.|
#  /___-_-,|.\  \   /  /.|
#       // |''\.;   ;,/ '|
#       `==|:=         =:|
#          `.          .'
#            :-._____.-:
#           `''       `''

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def main():
    cap = cv2.VideoCapture(0)
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

            mouth_left_x = landmarks[mp_pose.PoseLandmark.LEFT_MOUTH.value].x
            mouth_left_y = landmarks[mp_pose.PoseLandmark.LEFT_MOUTH.value].y
            mouth_left_z = landmarks[mp_pose.PoseLandmark.LEFT_MOUTH.value].z 

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

            mouth_right_x = landmarks[mp_pose.PoseLandmark.RIGHT_MOUTH.value].x
            mouth_right_y = landmarks[mp_pose.PoseLandmark.RIGHT_MOUTH.value].y
            mouth_right_z = landmarks[mp_pose.PoseLandmark.RIGHT_MOUTH.value].z

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
            
            
        print(f"Coordenadas del hombro izquierdo - X: {shoulder_left_x}, Y: {shoulder_left_y}, Z: {shoulder_left_z}")

        cv2.imshow("Image",img)
        key = cv2.waitKey(1) 
        if key == 27:
            break

if __name__ == "__main__":
    main()