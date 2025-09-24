import os
import sys
import cv2
import time
import glob
import numpy as np
import face_recognition
from pathlib import Path

def load_known_faces(dataset_dir: Path):
    """
    Load encodings for each person. Expects structure:
      dataset_dir/person_name/*.jpg|*.png|*.jpeg
    Returns (encodings_list, names_list)
    """
    encodings = []
    names = []

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    # Iterate people
    for person_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        person_name = person_dir.name
        image_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            image_paths.extend(glob.glob(str(person_dir / ext)))

        if not image_paths:
            print(f"[WARN] No images for {person_name} in {person_dir}")
            continue

        print(f"[INFO] Processing {person_name} ({len(image_paths)} images)...")
        for img_path in image_paths:
            image = face_recognition.load_image_file(img_path)
            # Convert and optionally resize very large images for speed
            if max(image.shape[:2]) > 1600:
                scale = 1600 / max(image.shape[:2])
                image = cv2.resize(image, (int(image.shape[1]*scale), int(image.shape[0]*scale)))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image

            boxes = face_recognition.face_locations(image, model="hog")  # switch to "cnn" if you have CUDA
            if not boxes:
                print(f"[WARN] No face found in {img_path}, skipping.")
                continue

            # Some images may have multiple faces; take the first one
            encs = face_recognition.face_encodings(image, known_face_locations=boxes)
            if not encs:
                print(f"[WARN] Could not compute encoding for {img_path}, skipping.")
                continue

            encodings.append(encs[0])
            names.append(person_name)

    if not encodings:
        raise RuntimeError("No face encodings loaded. Check your dataset images.")
    print(f"[INFO] Loaded {len(encodings)} encodings for {len(set(names))} people.")
    return encodings, names

def annotate_frame(frame, face_locations, face_names):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 24), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return frame

def main():
    dataset_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dataset")
    tolerance = float(sys.argv[2]) if len(sys.argv) > 2 else 0.45  # lower = stricter
    model = "hog"  # change to "cnn" if dlib compiled with CUDA

    known_encodings, known_names = load_known_faces(dataset_dir)

    print("[INFO] Starting webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try a different index (1/2) or check permissions.")

    process_every_other = True
    process_this_frame = True

    fps_avg = None
    last_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame grab failed.")
                break

            # Resize to 1/4 for speed, then convert BGR->RGB
            small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            face_locations = []
            face_names = []

            if not process_every_other or process_this_frame:
                # Find faces
                face_locations = face_recognition.face_locations(rgb_small, model=model)
                # Encode faces
                encodings = face_recognition.face_encodings(rgb_small, face_locations)

                for face_encoding in encodings:
                    # Compare to known faces
                    matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=tolerance)
                    name = "Unknown"

                    # Use the best (smallest) distance
                    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                    if len(face_distances) > 0:
                        best_idx = np.argmin(face_distances)
                        if matches[best_idx]:
                            name = known_names[best_idx]

                    face_names.append(name)

            process_this_frame = not process_this_frame if process_every_other else True

            # Scale back up face locations since frame was scaled to 1/4 size
            scaled_locations = []
            for (top, right, bottom, left) in face_locations:
                scaled_locations.append((top*4, right*4, bottom*4, left*4))

            frame = annotate_frame(frame, scaled_locations, face_names)

            # FPS display
            now = time.time()
            fps = 1.0 / (now - last_time) if now > last_time else 0.0
            last_time = now
            fps_avg = 0.9 * fps_avg + 0.1 * fps if fps_avg is not None else fps
            cv2.putText(frame, f"FPS: {fps_avg:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("Face ID (q to quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    """
    Usage:
      python realtime_face_id.py                # uses ./dataset and tolerance=0.45
      python realtime_face_id.py /path/to/dataset 0.5
    """
    main()
