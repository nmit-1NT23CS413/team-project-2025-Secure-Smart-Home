import tensorflow as tf
try:
    interpreter = tf.lite.Interpreter(model_path="models/ssd_mobilenet_v2_coco.tflite")
    interpreter.allocate_tensors()
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)
