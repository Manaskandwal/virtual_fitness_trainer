import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Curl and shoulder press counter variables
curl_counter, shoulder_press_counter = 0, 0
curl_stage, shoulder_press_stage = None, None

# Feedback colors
CORRECT_COLOR = (0, 255, 0)   # Green for correct
WRONG_COLOR = (0, 0, 255)     # Red for incorrect

cap = cv2.VideoCapture(0)

# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Coordinates for shoulders, elbows, wrists, and hips
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            # Calculate angles
            elbow_angle = calculate_angle(shoulder, elbow, wrist)  # Bicep curl
            shoulder_press_angle = calculate_angle(elbow, shoulder, hip)  # Shoulder press

            # Set default color to wrong
            feedback_color = WRONG_COLOR

            # **Bicep Curl Detection**
            if elbow_angle > 160:  # Arm fully extended
                curl_stage = "down"
            if elbow_angle < 45 and curl_stage == "down":  # Arm flexed
                curl_stage = "up"
                curl_counter += 1
                print(f"Curl Reps: {curl_counter}")
                feedback_color = CORRECT_COLOR  # Green if rep is complete

            # **Shoulder Press Detection**
            if shoulder_press_angle > 160:  # Arms down (rest position)
                shoulder_press_stage = "down"
            if shoulder_press_angle < 90 and shoulder_press_stage == "down":  # Arms raised overhead
                shoulder_press_stage = "up"
                shoulder_press_counter += 1
                print(f"Shoulder Press Reps: {shoulder_press_counter}")
                feedback_color = CORRECT_COLOR  # Green if rep is complete

            # Show joint angles on screen for reference
            cv2.putText(image, f'Elbow Angle: {int(elbow_angle)}', (10, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, feedback_color, 2, cv2.LINE_AA)
            cv2.putText(image, f'Shoulder Angle: {int(shoulder_press_angle)}', (10, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, feedback_color, 2, cv2.LINE_AA)

            # Show visual feedback with proper spacing
            cv2.putText(image, 'BICEP CURLS', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, feedback_color, 2, cv2.LINE_AA)
            cv2.putText(image, f'{curl_counter}', (350, 50),  # Move the counter away
                        cv2.FONT_HERSHEY_SIMPLEX, 1, feedback_color, 2, cv2.LINE_AA)

            cv2.putText(image, 'SHOULDER PRESS', (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, feedback_color, 2, cv2.LINE_AA)
            cv2.putText(image, f'{shoulder_press_counter}', (350, 100),  # Move the counter away
                        cv2.FONT_HERSHEY_SIMPLEX, 1, feedback_color, 2, cv2.LINE_AA)

        except:
            pass

        # Render detections with feedback
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=feedback_color, thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=feedback_color, thickness=2, circle_radius=2))

        cv2.imshow('Exercise Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

