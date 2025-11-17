# tflite_helper.py
import numpy as np
import cv2
import tensorflow as tf

class TFLiteDetector:
    def __init__(self, model_path, input_size=(300, 300), threshold=0.5):
        # Load the TensorFlow Lite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Store configuration
        self.size = input_size
        self.threshold = threshold

    def preprocess(self, frame):
        """Resize and prepare frame according to model requirements."""
        img = cv2.resize(frame, self.size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)

        # Convert to correct dtype based on model input
        if self.input_details[0]['dtype'] == np.uint8:
            img = img.astype(np.uint8)  # Quantized model (0–255)
        else:
            img = (img / 255.0).astype(np.float32)  # Float model (0–1)
        return img

    def detect(self, frame):
        """Run object detection on a single frame."""
        inp = self.preprocess(frame)
        self.interpreter.set_tensor(self.input_details[0]['index'], inp)
        self.interpreter.invoke()

        # Extract detection results
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]

        detections = []
        for box, cls, sc in zip(boxes, classes, scores):
            if sc > self.threshold:
                detections.append({
                    "class_id": int(cls),
                    "score": float(sc),
                    "box": box.tolist()
                })

        return detections
