import cv2
import mediapipe as mp 
import pandas as pd

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Dataframe
columns = []
for i in range(33):
    columns.extend([f'X_{i}', f'Y_{i}', f'Z_{i}'])

df = pd.DataFrame(columns=columns)

def update_dataframe(landmarks):
    global df
    new_row = {}
    for i, landmark in enumerate(landmarks):
        new_row[f'X_{i}'] = landmark.x
        new_row[f'Y_{i}'] = landmark.y
        new_row[f'Z_{i}'] = landmark.z if landmark.HasField('z') else None
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

def main():
    file_name = 'Reference 2.mp4'
    cap = cv2.VideoCapture(file_name)
    while cap.isOpened(): 
        success, img = cap.read()
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(frame_rgb)
        mp_drawing.draw_landmarks(img, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2), connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
        
        # Verificar si se detectaron poses
        if pose_results.pose_landmarks:
            # Acceder a las coordenadas de los puntos de referencia
            landmarks = pose_results.pose_landmarks.landmark

            update_dataframe(landmarks)

        cv2.imshow("Image", img)
        key = cv2.waitKey(1) 
        if key == 27:
            break
    
    df.to_csv('Lets_Train.csv', index=False)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
