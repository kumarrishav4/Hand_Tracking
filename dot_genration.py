import cv2
import mediapipe as mp

def plot_points(image, points):
    for point in points:
        x, y = point
        cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)

def capture_and_track():
    cap = cv2.VideoCapture(0)  

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    cv2.namedWindow("Fingertip Tracking", cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                fingertips = [
                    (int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0]))
                    for lm in hand_landmarks.landmark
                ]
                plot_points(frame, fingertips)

        cv2.imshow("Fingertip Tracking", frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_track()
