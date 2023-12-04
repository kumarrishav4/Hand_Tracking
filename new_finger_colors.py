import cv2
import mediapipe as mp

def lines_and_box(image, points):
    finger_colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 255, 255)]
    palm_color = (255, 0, 255)


    fingers = [[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16], [17,18,19,20]]
    for i, finger in enumerate(fingers):
        color = finger_colors[i]
        for j in range(len(finger) - 1):
            point1 = points[finger[j]]
            point2 = points[finger[j + 1]]
            cv2.line(image, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])), color, 2)


    palm_box = [points[i] for i in [9,5,1,0,17,13,9]] 
    for i in range(len(palm_box) - 1):
        cv2.line(image, (int(palm_box[i][0]), int(palm_box[i][1])), (int(palm_box[i + 1][0]), int(palm_box[i + 1][1])), palm_color, 2)

def capture_and_track():
    cap = cv2.VideoCapture(0)  

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    cv2.namedWindow("Hand Tracking with Colored Box", cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = [
                    (int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0]))
                    for lm in hand_landmarks.landmark
                ]
                lines_and_box(frame, landmarks)

        cv2.imshow("Hand Tracking with Colored Box", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_track()
