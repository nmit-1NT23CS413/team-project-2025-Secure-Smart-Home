# main.py
from sensor_sim import simulate_pir
from alert_system import send_alert
from face_recog import verify_face
from tflite_helper import TFLiteDetector
import cv2

detector = TFLiteDetector("models/ssd_mobilenet_v2_coco.tflite")

def main():
    print("System armed. Monitoring...")
    for motion in simulate_pir(2.0):
        if motion:
            send_alert("Motion", "PIR triggered!")
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            if not ret: continue
            dets = detector.detect(frame)
            person = any(d["class_id"] == 0 for d in dets)  # class 0 ≈ person
            if person:
                same = verify_face("faces/authorized/owner.jpg", "faces/test/current.jpg")
                if same:
                    send_alert("Authorized", "Known face detected")
                else:
                    send_alert("Intrusion", "Unknown face detected!")
            else:
                send_alert("Noise", "No person detected")

if __name__ == "__main__":
    main()
