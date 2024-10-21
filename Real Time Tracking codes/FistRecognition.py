import cv2
import mediapipe as mp
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import math
import os


def calculate_angle(p1, p2, p3):
    a = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    b = math.sqrt((p2[0] - p3[0]) ** 2 + (p2[1] - p3[1]) ** 2)
    c = math.sqrt((p3[0] - p1[0]) ** 2 + (p3[1] - p1[1]) ** 2)
    angle = math.degrees(math.acos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b)))
    return angle

def is_fist(landmarks):
    finger_tips = [landmarks[i] for i in [4, 8, 12, 16, 20]]  
    palm_base = landmarks[0]  
    fist_threshold = 40  

    distances = [math.sqrt((tip[0] - palm_base[0]) ** 2 + (tip[1] - palm_base[1]) ** 2) for tip in finger_tips]
    
    return all(dist < fist_threshold for dist in distances)

def lines_and_box(image, points):
    fingers = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
    angles = []
    
    if is_fist(points):
        cv2.putText(image, 'Fist Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        for i, finger in enumerate(fingers):
            for j in range(len(finger) - 2):
                p1 = points[finger[j]]
                p2 = points[finger[j + 1]]
                p3 = points[finger[j + 2]]
                angle = calculate_angle(p1, p2, p3)
                angles.append(angle)
                cv2.putText(image, f'{angle:.1f}', (p2[0], p2[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            for j in range(len(finger) - 1):
                p1 = points[finger[j]]
                p2 = points[finger[j + 1]]
                cv2.line(image, p1, p2, (0, 255, 0), 2)
    
    return angles

class HandTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Tracker")
        
        self.start_button = tk.Button(root, text="Start", command=self.start_tracking)
        self.start_button.pack(pady=10)
        
        self.stop_button = tk.Button(root, text="Stop", command=self.stop_tracking, state=tk.DISABLED)
        self.stop_button.pack(pady=10)
        
        self.cap = None
        self.csv_data = []
        self.running = False

    def start_tracking(self):
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.running = True
        self.capture_and_track()

    def stop_tracking(self):
        self.running = False
        self.stop_button.config(state=tk.DISABLED)
        self.start_button.config(state=tk.NORMAL)
        
        self.save_to_csv()

    def capture_and_track(self):
        self.cap = cv2.VideoCapture(0)
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands()
        
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
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
                    
                    if landmarks and len(landmarks) >= 21:
                        angles = lines_and_box(frame, landmarks)
                        self.csv_data.append(angles)
                    else:
                        print("No valid landmarks detected or landmark data is incomplete.")
            
            cv2.imshow("Hand Tracking with Box and Angles", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def save_to_csv(self):
        if self.csv_data:
            df = pd.DataFrame(self.csv_data, columns=[f'Finger_{i}_Joint_{j}_Angle' for i in range(1, 6) for j in range(1, 4)])
            file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if file_path:
                df.to_csv(file_path, index=False)
            else:
                print("Save operation canceled.")
        else:
            print("No data to save.")

if __name__ == "__main__":
    root = tk.Tk()
    app = HandTrackerApp(root)
    root.mainloop()
