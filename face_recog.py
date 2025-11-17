# face_recog.py
import os
from deepface import DeepFace

def verify_face(known, unknown):
    """
    Compare two face images and return True if they belong to the same person, else False.
    Automatically handles missing files and errors.
    """
    # Check if both images exist
    if not os.path.exists(known):
        print(f"❌ Error: Authorized face not found at {known}")
        return False

    if not os.path.exists(unknown):
        print(f"❌ Error: Test face not found at {unknown}")
        return False

    try:
        result = DeepFace.verify(
            img1_path=known,
            img2_path=unknown,
            enforce_detection=False  # skip crash if no face detected
        )
        is_verified = result.get("verified", False)
        distance = result.get("distance", None)
        model = result.get("model", "VGG-Face")

        if is_verified:
            print(f"✅ Match confirmed! ({model}, distance={distance:.4f})")
        else:
            print(f"⚠️ No match. ({model}, distance={distance:.4f})")

        return is_verified

    except Exception as e:
        print(f"⚠️ DeepFace verification failed: {e}")
        return False


if __name__ == "__main__":
    # Test the face verification manually
    authorized_path = "faces/authorized/owner.jpg"
    test_path = "faces/test/intruder.jpg"

    print("🔍 Running test face verification...")
    result = verify_face(authorized_path, test_path)
    print(f"Verification result: {result}")
