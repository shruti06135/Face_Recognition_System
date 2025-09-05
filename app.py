import cv2
import os
import pickle
import time
import shutil
import tempfile
import math
from deepface import DeepFace
import numpy as np
import dlib
from scipy.spatial import distance

# --- Constants and Directory Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "user_data.pkl")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
# Make sure you download this file and place it in the same directory
PREDICTOR_PATH = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat") 

os.makedirs(DATASET_DIR, exist_ok=True)

# --- Pre-load Models for Computational Efficiency ---
print("🚀 Initializing system and pre-loading models...")
try:
    FACE_DETECTOR = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    dlib_detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    
    # Warm up DeepFace FaceNet model to avoid lag on first run
    _ = DeepFace.represent(
        img_path=np.zeros((100, 100, 3), dtype=np.uint8),
        model_name="Facenet",
        detector_backend='skip'
    )
    print("✅ All models initialized successfully.")
except Exception as e:
    print(f"❌ Critical initialization error: {e}")
    exit()

# --- Bullet 2: Liveness Detection Helpers (EAR Logic) ---
def eye_aspect_ratio(eye):
    """Calculate the Eye Aspect Ratio (EAR) to determine open/closed state."""
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def check_liveness(frame):
    """Returns a tuple: (blink_detected, eyes_are_open)"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = dlib_detector(gray, 0)
    
    for rect in rects:
        shape = predictor(gray, rect)
        shape = np.array([[p.x, p.y] for p in shape.parts()])
        
        left_eye = shape[36:42]
        right_eye = shape[42:48]
        
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        
        # Thresholds: EAR < 0.21 indicates a closed eye/blink
        if ear < 0.21:
            return True, False
        return False, True
        
    return False, False

# --- Bullet 4: Image Preprocessing Layer (CLAHE Enhancement) ---
def preprocess_frame(frame):
    """Enhance facial features under varying lighting conditions using CLAHE."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

# --- Bullet 3 & 5: Multi-Image Sign-Up Capture Engine ---
def capture_signup_dataset(username, num_images=10):
    cap = cv2.VideoCapture(0)
    user_dir = os.path.join(DATASET_DIR, username)
    os.makedirs(user_dir, exist_ok=True)
    
    images_captured = 0
    print(f"\n📸 Initializing Registration Pipeline for: {username}")
    
    while images_captured < num_images:
        ret, frame = cap.read()
        if not ret: break

        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_DETECTOR.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        feedback = "Position your face clearly inside the camera view"
        box_color = (0, 0, 255) 

        if len(faces) > 1:
            feedback = "Multiple faces detected! Please ensure only one person is visible."
        elif len(faces) == 1:
            x, y, w, h = faces[0]
            feedback = "Face Detected! Capturing dataset..."
            box_color = (0, 255, 0)
            
            images_captured += 1
            img_path = os.path.join(user_dir, f"{images_captured}.jpg")
            
            # Apply processing before file commit
            processed = preprocess_frame(frame)
            cv2.imwrite(img_path, processed)
            
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), box_color, 2)
            cv2.putText(display_frame, "CAPTURED!", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Registration Dashboard", display_frame)
            cv2.waitKey(300)
            continue
            
        cv2.putText(display_frame, feedback, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
        cv2.imshow("Registration Dashboard", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            shutil.rmtree(user_dir)
            cap.release()
            cv2.destroyAllWindows()
            return None

    cap.release()
    cv2.destroyAllWindows()
    return user_dir

# --- Bullet 3 & 5: Multi-Image Login Verification Engine ---
def capture_login_pipeline(num_images=10):
    cap = cv2.VideoCapture(0)
    temp_dir = tempfile.mkdtemp()
    captured_paths = []
    
    images_captured = 0
    start_time = time.time()
    timeout = 25
    blink_verified = False
    
    print("\n🔐 Initializing Secure Multi-Frame Login Window...")
    
    while images_captured < num_images and (time.time() - start_time) < timeout:
        ret, frame = cap.read()
        if not ret: break

        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_DETECTOR.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        if len(faces) == 1:
            x, y, w, h = faces[0]
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            
            blink, eyes_open = check_liveness(frame)
            
            if blink:
                blink_verified = True
                
            if blink_verified:
                cv2.putText(display_frame, "LIVENESS: VERIFIED ✅", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "LIVENESS: PLEASE BLINK 👁", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Ensure we capture high quality frames with eyes wide open post-blink validation
            if eyes_open and blink_verified:
                images_captured += 1
                img_path = os.path.join(temp_dir, f"verification_{images_captured}.jpg")
                processed = preprocess_frame(frame)
                cv2.imwrite(img_path, processed)
                captured_paths.append(img_path)
                
                cv2.putText(display_frame, f"ANALYZING FRAME {images_captured}/{num_images}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow("Authentication Terminal", display_frame)
                cv2.waitKey(150)
                continue

        cv2.imshow("Authentication Terminal", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    cap.release()
    cv2.destroyAllWindows()
    return captured_paths, temp_dir, blink_verified

# --- Main Authentication Interfaces ---
def sign_up(username):
    if not username:
        print("❌ Username validation failed: Field cannot be empty.")
        return

    db = {}
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "rb") as f:
            db = pickle.load(f)

    if username in db:
        print(f"❌ Conflict: User identity '{username}' already occupies registry footprint.")
        return

    user_dir_path = capture_signup_dataset(username)
    if not user_dir_path:
        print("❌ Pipeline Abortion: Registration process dismissed by user.")
        return

    db[username] = user_dir_path
    with open(DATA_FILE, "wb") as f:
        pickle.dump(db, f)
    print(f"✅ Identity Registry Complete for record: '{username}'")

def login(username):
    if not os.path.exists(DATA_FILE):
        print("❌ Verification halted: System registry database is currently empty.")
        return

    with open(DATA_FILE, "rb") as f:
        db = pickle.load(f)

    if username not in db:
        print(f"❌ Access Denied: User '{username}' records not found.")
        return

    registered_dir = db[username]
    registered_files = os.listdir(registered_dir)
    if not registered_files:
        print("❌ Integrity Error: User folder records missing or corrupted.")
        return
        
    # Pick baseline registration image
    base_reference_img = os.path.join(registered_dir, registered_files[0])
    
    login_frames, temp_dir, liveness_passed = capture_login_pipeline()
    
    if not liveness_passed:
        print("❌ Security Alert: Passive anti-spoofing liveness verification failed. Access Denied.")
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        return

    if not login_frames:
        print("❌ Session Timeout: Failed to gather reliable frame validation matrix.")
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        return

    print("🔄 Processing Multi-Frame Deep Learning verification matrix via FaceNet...")
    successful_matches = 0
    
    try:
        for path in login_frames:
            try:
                # Bullet 1: Match execution via DeepFace using FaceNet core weights
                result = DeepFace.verify(
                    img1_path=base_reference_img,
                    img2_path=path,
                    model_name="Facenet",
                    enforce_detection=True
                )
                if result["verified"]:
                    successful_matches += 1
            except Exception:
                continue
    finally:
        shutil.rmtree(temp_dir)

    # Establish an evaluation threshold (minimum 50% matching multi-frame matrix passes)
    min_required_threshold = math.ceil(len(login_frames) * 0.5)
    print(f"📊 Assessment Summary: Passed [{successful_matches}/{len(login_frames)}] frame validation thresholds.")
    
    if successful_matches >= min_required_threshold:
        print("🔓 [SUCCESS] Multi-Factor Biometric Authentication Passed. Welcome back.")
    else:
        print("🔒 [FAILURE] Spatial Authentication Failed. Signature variance exceeded limit.")

# --- Interactive System Shell Context ---
if __name__ == "__main__":
    while True:
        print("\n=== Biometric Cryptographic Access Shell ===")
        print("1. Initialize New Profile Enrolment (Sign Up)")
        print("2. Request System Entry Gateway (Login)")
        print("3. Terminate Session Environment")
        user_choice = input("Select Execution Route (1/2/3): ").strip()

        if user_choice == "1":
            name_input = input("Assign identity name tag: ").strip()
            sign_up(name_input)
        elif user_choice == "2":
            name_input = input("Provide profile verification key: ").strip()
            login(name_input)
        elif user_choice == "3":
            print("👋 Session Terminated Safely.")
            break
        else:
            print("❌ Input parsing mismatch. Try again.")