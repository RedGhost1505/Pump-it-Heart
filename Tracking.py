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
            
        print(f"Coordenadas del hombro izquierdo - X: {shoulder_left_x}, Y: {shoulder_left_y}, Z: {shoulder_left_z}")

        cv2.imshow("Image",img)
        key = cv2.waitKey(1) 
        if key == 27:
            break

if __name__ == "__main__":
    main()