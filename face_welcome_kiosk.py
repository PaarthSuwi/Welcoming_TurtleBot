import os
import sys
import cv2
import time
import glob
import numpy as np
import face_recognition
from pathlib import Path
from typing import Dict, Tuple, List

WELCOME_TEXT = "Welcome to symbitech2025"
WELCOME_SECONDS = 4.0        # how long to show the welcome screen
COOLDOWN_SECONDS = 10.0       # per-person cooldown to avoid repeat triggers
RECOG_TOLERANCE = 0.45        # lower = stricter; 0.45–0.5 works well

def pick_profile_image(image_paths: List[str]) -> str:
    """Pick a representative image (first by default)."""
    return sorted(image_paths)[0]

def load_known_faces(dataset_dir: Path) -> Tuple[List[np.ndarray], List[str], Dict[str, str]]:
    """
    Loads face encodings + maps each person to a representative image path.
    Expects: dataset_dir/person_name/*.jpg|*.png|...
    Returns: (encodings, names, profile_image_for_person)
    """
    encodings, names = [], []
    profile_images: Dict[str, str] = {}

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    for person_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()]):
        person_name = person_dir.name
        image_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            image_paths.extend(glob.glob(str(person_dir / ext)))
        if not image_paths:
            print(f"[WARN] No images for {person_name} in {person_dir}")
            continue

        # Save a profile image path for the welcome screen
        profile_images[person_name] = pick_profile_image(image_paths)

        print(f"[INFO] Processing {person_name} ({len(image_paths)} images)...")
        for img_path in image_paths:
            image = face_recognition.load_image_file(img_path)
            # Resize very large images for speed/robustness
            if max(image.shape[:2]) > 1600:
                scale = 1600 / max(image.shape[:2])
                image = cv2.resize(image, (int(image.shape[1]*scale), int(image.shape[0]*scale)))

            boxes = face_recognition.face_locations(image, model="hog")
            if not boxes:
                print(f"[WARN] No face in {img_path}, skipping.")
                continue

            encs = face_recognition.face_encodings(image, known_face_locations=boxes)
            if not encs:
                print(f"[WARN] Could not compute encoding for {img_path}, skipping.")
                continue

            encodings.append(encs[0])
            names.append(person_name)

    if not encodings:
        raise RuntimeError("No face encodings loaded. Check your dataset images.")
    print(f"[INFO] Loaded {len(encodings)} encodings for {len(set(names))} people.")
    return encodings, names, profile_images

def draw_label(frame, box, text):
    (top, right, bottom, left) = box
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    label_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (left, bottom - 28), (left + label_size[0] + 12, bottom), (0, 255, 0), -1)
    cv2.putText(frame, text, (left + 6, bottom - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

def make_welcome_canvas(person_name: str, profile_path: str, screen_size=(900, 600)) -> np.ndarray:
    """
    Create a nice welcome image with person photo + text.
    """
    w, h = screen_size[0], screen_size[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Background gradient
    for y in range(h):
        alpha = y / max(1, h - 1)
        color_top = np.array([30, 30, 30], dtype=np.float32)
        color_bottom = np.array([70, 130, 180], dtype=np.float32)  # steel-ish blue
        color = (1 - alpha) * color_top + alpha * color_bottom
        canvas[y, :, :] = color.astype(np.uint8)

    # Load and place profile image
    if profile_path and Path(profile_path).exists():
        img = cv2.imread(profile_path)
        if img is not None:
            # Make it square-ish and large, keep aspect
            target_h = int(h * 0.65)
            scale = target_h / img.shape[0]
            new_w = int(img.shape[1] * scale)
            img = cv2.resize(img, (new_w, target_h))
            # Convert to centered card with rounded-ish border (simple rectangle)
            card_w = min(int(w * 0.42), img.shape[1] + 40)
            card_h = img.shape[0] + 40
            card = np.full((card_h, card_w, 3), 240, dtype=np.uint8)

            # paste image in center of card
            x_offset = (card_w - img.shape[1]) // 2
            y_offset = (card_h - img.shape[0]) // 2
            card[y_offset:y_offset + img.shape[0], x_offset:x_offset + img.shape[1]] = img

            # drop shadow
            shadow = card.copy()
            cv2.GaussianBlur(shadow, (0, 0), 9, dst=shadow)
            sx = int(w * 0.09)
            sy = int(h * 0.14)
            canvas[sy + 8:sy + 8 + card_h, sx + 8:sx + 8 + card_w] = cv2.addWeighted(
                canvas[sy + 8:sy + 8 + card_h, sx + 8:sx + 8 + card_w], 0.5, shadow, 0.5, 0
            )
            # card
            canvas[sy:sy + card_h, sx:sx + card_w] = card

    # Right-side text block
    right_x = int(w * 0.55)
    top_y = int(h * 0.3)

    # Headline
    title = WELCOME_TEXT
    cv2.putText(canvas, title, (right_x, top_y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)

    # Subheadline
    sub = f"Hello, {person_name}!"
    cv2.putText(canvas, sub, (right_x, top_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Footer/CTA
    footer = "We’re glad you’re here."
    cv2.putText(canvas, footer, (right_x, top_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 230, 230), 2)

    return canvas

def main():
    dataset_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dataset")
    tolerance = float(sys.argv[2]) if len(sys.argv) > 2 else RECOG_TOLERANCE
    det_model = "hog"  # set to "cnn" if you have dlib+CUDA

    known_encodings, known_names, profile_images = load_known_faces(dataset_dir)

    print("[INFO] Starting webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try a different index (1/2) or check permissions.")

    process_every_other = True
    process_this_frame = True
    last_seen_time: Dict[str, float] = {}
    showing_welcome_until = 0.0
    current_welcome_name = None
    welcome_win = "Welcome"
    camera_win = "Face ID"

    try:
        while True:
            now = time.time()

            # If currently showing welcome screen, keep it until time elapses
            if now < showing_welcome_until and current_welcome_name:
                # We keep the welcome window alive
                if cv2.getWindowProperty(welcome_win, cv2.WND_PROP_VISIBLE) < 1:
                    cv2.namedWindow(welcome_win, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(welcome_win, 1000, 700)
                profile_path = profile_images.get(current_welcome_name, "")
                canvas = make_welcome_canvas(current_welcome_name, profile_path, (1000, 700))
                cv2.imshow(welcome_win, canvas)

                # Still poll keyboard
                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    break
                # Skip grabbing frames while welcome is up
                continue
            else:
                # If the welcome screen time is over, close it
                if current_welcome_name:
                    try:
                        cv2.destroyWindow(welcome_win)
                    except:
                        pass
                    current_welcome_name = None

            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame grab failed.")
                break

            # Prepare small RGB frame for detection/encoding
            small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            face_locations = []
            face_names = []

            if not process_every_other or process_this_frame:
                face_locations = face_recognition.face_locations(rgb_small, model=det_model)
                encodings = face_recognition.face_encodings(rgb_small, face_locations)

                for face_encoding in encodings:
                    matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=tolerance)
                    name = "Unknown"

                    distances = face_recognition.face_distance(known_encodings, face_encoding)
                    if len(distances) > 0:
                        best_idx = np.argmin(distances)
                        if matches[best_idx]:
                            name = known_names[best_idx]

                    face_names.append(name)

            process_this_frame = not process_this_frame if process_every_other else True

            # Scale back boxes and draw
            scaled_boxes = []
            for (top, right, bottom, left) in face_locations:
                scaled_boxes.append((top * 4, right * 4, bottom * 4, left * 4))

            if cv2.getWindowProperty(camera_win, cv2.WND_PROP_VISIBLE) < 1:
                cv2.namedWindow(camera_win, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(camera_win, 960, 540)

            for box, name in zip(scaled_boxes, face_names):
                draw_label(frame, box, name)

            cv2.imshow(camera_win, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            # Trigger welcome screen if: known face AND not in cooldown
            for name in face_names:
                if name == "Unknown":
                    continue
                last = last_seen_time.get(name, 0.0)
                if (time.time() - last) > COOLDOWN_SECONDS:
                    # Open/refresh welcome
                    current_welcome_name = name
                    showing_welcome_until = time.time() + WELCOME_SECONDS
                    last_seen_time[name] = time.time()
                    # Create window immediately for responsiveness
                    cv2.namedWindow(welcome_win, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(welcome_win, 1000, 700)
                    profile_path = profile_images.get(name, "")
                    canvas = make_welcome_canvas(name, profile_path, (1000, 700))
                    cv2.imshow(welcome_win, canvas)
                    cv2.waitKey(10)  # draw once right away
                    break  # only welcome the first recognized name per frame

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    """
    Usage:
      python face_welcome_kiosk.py                      # uses ./dataset
      python face_welcome_kiosk.py /path/to/dataset     # custom dataset
      python face_welcome_kiosk.py ./dataset 0.5        # custom tolerance
    """
    dataset_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dataset")
    # basic sanity check before running
    if not dataset_dir.exists():
        print(f"[ERROR] Dataset directory not found: {dataset_dir}")
        sys.exit(1)
    main()
