import cv2
from tflite_helper import TFLiteDetector

model_path = "models/ssd_mobilenet_v2_coco.tflite"   # download this model
detector = TFLiteDetector(model_path)

cap = cv2.VideoCapture(0)
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret: break
    dets = detector.detect(frame)
    for d in dets:
        if d["score"] > 0.5:
            cv2.putText(frame, f"ID:{d['class_id']} {d['score']:.2f}",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release(); cv2.destroyAllWindows()
