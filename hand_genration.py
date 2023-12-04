import cv2
import mediapipe as mp

def lines_and_box(image, points):

    for point in points:
        x, y = point
        cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)


    fingers = [[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16], [17,18,19,20]]
    for finger in fingers:
        for i in range(len(finger) - 1):
            point1 = points[finger[i]]
            point2 = points[finger[i + 1]]
            cv2.line(image, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])), (0, 255, 0), 2)

    palm_box = [points[i] for i in [9,5,1,0,17,13,9] ]
    for i in range(len(palm_box) - 1):
        cv2.line(image, (int(palm_box[i][0]), int(palm_box[i][1])), (int(palm_box[i + 1][0]), int(palm_box[i + 1][1])), (0, 255, 0), 2)
    cv2.line(image, (int(palm_box[-1][0]), int(palm_box[-1][1])), (int(palm_box[0][0]), int(palm_box[0][1])), (0, 255, 0), 2)

def capture_and_track():
    cap = cv2.VideoCapture(0)  

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    cv2.namedWindow("Hand Tracking with Box", cv2.WINDOW_NORMAL)

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

        cv2.imshow("Hand Tracking with Box", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_track()
