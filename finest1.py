import os, sys, time, glob, json, queue, threading, platform
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import cv2
import face_recognition

import tkinter as tk
from PIL import Image, ImageTk, ImageDraw, ImageOps, ImageFont, Image
import pyttsx3
import qrcode

# -------- NumPy 1.x / 2.x string dtype shim --------
try:
    STR_DTYPE = np.unicode_
except AttributeError:
    STR_DTYPE = np.str_

# ==============================
# Config
# ==============================
APP_TITLE = "SymbiTech 2025"
WELCOME_LINE = "WELCOME ({name}) hope you enjoy SYMBITECH25"
RECOG_TOLERANCE = 0.45

WELCOME_SECONDS = 10.0                 # banner stays up this long after each greet

# Speak this often per guest while they remain recognized.
# Set to 0.0 if you want speech literally on *every frame* of recognition.
SPEAK_EVERY_SECONDS = 1.0

# Bigger ROI (centered)
ROI_REL = (0.20, 0.10, 0.80, 0.90)     # (x1, y1, x2, y2) in [0..1]

# Recognition settings
DET_MODEL = "hog"                       # or "cnn" (requires CUDA-enabled dlib)
CAM_INDEX = 0
CACHE_FILE = "encodings_cache.npz"
GUESTS_FILE = "guests.json"             # {"Name":{"title":"...","organization":"...","linkedin":"..."}}

# Performance / quality
PROCESS_EVERY_N_FRAMES = 1              # run recognizer every frame
DOWNSCALE = 0.5                         # 0.5 = more detail than 0.25
UPSAMPLE = 1                            # helps farther faces on HOG
TARGET_CAM_W, TARGET_CAM_H = 1280, 720
MAX_BOX_AGE_FRAMES = 10

# Adaptive sweep: if nothing found for N processed frames, scan full frame
NO_DETECT_FRAMES_FOR_SWEEP = 15

# Optional: force a brand font
FORCE_FONT_PATH = None  # e.g. "C:/Windows/Fonts/seguisb.ttf"

# ==============================
# Dataset & Encodings
# ==============================
def pick_profile_image(image_paths: List[str]) -> str:
    return sorted(image_paths)[0]

def scan_dataset(dataset_dir: Path) -> Dict[str, List[str]]:
    people_images = {}
    for person_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()]):
        paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            paths.extend(glob.glob(str(person_dir / ext)))
        if paths:
            people_images[person_dir.name] = paths
    return people_images

def _as_float_matrix(enc_list: List[np.ndarray]) -> np.ndarray:
    if len(enc_list) == 0:
        return np.empty((0, 128), dtype=np.float32)
    return np.vstack([np.asarray(e, dtype=np.float32) for e in enc_list]).astype(np.float32)

def load_or_build_encodings(dataset_dir: Path):
    if Path(CACHE_FILE).exists():
        try:
            data = np.load(CACHE_FILE, allow_pickle=False)
            enc = data["encodings"].astype(np.float32)
            names = data["names"].astype(str).tolist()
            k = data["profile_keys"].astype(str).tolist()
            v = data["profile_vals"].astype(str).tolist()
            profiles = dict(zip(k, v))
            if enc.ndim == 2 and enc.shape[1] == 128:
                return enc, names, profiles
        except Exception:
            pass

    people_images = scan_dataset(dataset_dir)
    enc_list, names = [], []
    profiles = {}
    for person, imgs in people_images.items():
        profiles[person] = pick_profile_image(imgs)
        for p in imgs:
            img = face_recognition.load_image_file(p)
            if max(img.shape[:2]) > 1600:
                s = 1600 / max(img.shape[:2])
                img = cv2.resize(img, (int(img.shape[1]*s), int(img.shape[0]*s)))
            boxes = face_recognition.face_locations(img, model=DET_MODEL)
            if not boxes:
                continue
            encs = face_recognition.face_encodings(img, boxes)
            if encs:
                enc_list.append(np.asarray(encs[0], dtype=np.float32))
                names.append(person)

    if not enc_list:
        raise RuntimeError("No face encodings loaded—check your dataset images.")

    enc = _as_float_matrix(enc_list)
    np.savez_compressed(
        CACHE_FILE,
        encodings=enc,
        names=np.array(names, dtype=STR_DTYPE),
        profile_keys=np.array(list(profiles.keys()), dtype=STR_DTYPE),
        profile_vals=np.array(list(profiles.values()), dtype=STR_DTYPE),
    )
    return enc, names, profiles

def load_guests_info(path: Path) -> Dict[str, Dict[str, str]]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# ==============================
# Threads: Camera, Recognizer, Audio
# ==============================
class CameraGrabber(threading.Thread):
    def __init__(self, index=0, width=1280, height=720):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.lock = threading.Lock()
        self.running = True
        self.latest = None

    def run(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.latest = frame
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            return None if self.latest is None else self.latest.copy()

    def stop(self):
        self.running = False
        time.sleep(0.05)
        try: self.cap.release()
        except: pass

class Recognizer(threading.Thread):
    """
    Detect/recognize on ROI every frame. Full-frame sweep after brief misses.
    """
    def __init__(self, cam: CameraGrabber, known_enc: np.ndarray, known_names: List[str]):
        super().__init__(daemon=True)
        self.cam = cam
        self.known_enc = known_enc
        self.known_names = known_names
        self.running = True
        self.out_lock = threading.Lock()
        self.last_boxes: List[Tuple[int,int,int,int]] = []
        self.last_names: List[str] = []
        self.box_age = 0
        self.missed = 0

    def _prep_for_detection(self, bgr):
        try:
            ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            y = clahe.apply(y)
            ycrcb = cv2.merge((y, cr, cb))
            bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        except Exception:
            pass
        return bgr

    def run(self):
        while self.running:
            frame = self.cam.read()
            if frame is None:
                time.sleep(0.01); continue

            H, W = frame.shape[:2]
            use_full = self.missed >= NO_DETECT_FRAMES_FOR_SWEEP
            if use_full:
                x1, y1, x2, y2 = 0, 0, W, H
            else:
                x1 = int(W * ROI_REL[0]); y1 = int(H * ROI_REL[1])
                x2 = int(W * ROI_REL[2]); y2 = int(H * ROI_REL[3])

            roi = frame[y1:y2, x1:x2]
            updated = False
            if roi.size > 0:
                roi = self._prep_for_detection(roi)
                small = cv2.resize(roi, (0, 0), fx=DOWNSCALE, fy=DOWNSCALE)
                rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

                locs = face_recognition.face_locations(
                    rgb_small, model=DET_MODEL, number_of_times_to_upsample=UPSAMPLE
                )
                encs = face_recognition.face_encodings(rgb_small, locs)

                names = []
                for e in encs:
                    e = np.asarray(e, dtype=np.float32)
                    matches = face_recognition.compare_faces(self.known_enc, e, tolerance=RECOG_TOLERANCE)
                    dists = face_recognition.face_distance(self.known_enc, e)
                    nm = "Unknown"
                    if len(dists):
                        i = int(np.argmin(dists))
                        if matches[i]:
                            nm = self.known_names[i]
                    names.append(nm)

                inv = 1.0 / DOWNSCALE
                boxes = []
                for (t, r, b, l) in locs:
                    top = int(y1 + t * inv)
                    right = int(x1 + r * inv)
                    bottom = int(y1 + b * inv)
                    left = int(x1 + l * inv)
                    boxes.append((top, right, bottom, left))

                with self.out_lock:
                    self.last_boxes = boxes
                    self.last_names = names
                    self.box_age = 0
                    updated = True

                if len(names) == 0:
                    self.missed += 1
                else:
                    self.missed = 0

            if not updated:
                with self.out_lock:
                    self.box_age += 1
                    if self.box_age > MAX_BOX_AGE_FRAMES:
                        self.last_boxes, self.last_names = [], []
                self.missed += 1

            time.sleep(0.001)

    def get_last(self):
        with self.out_lock:
            return list(self.last_boxes), list(self.last_names)

    def stop(self):
        self.running = False

class AudioGreeter(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.q = queue.Queue()  # items: (msg, delay_seconds)
        self.engine = pyttsx3.init()
        self.running = True

    def run(self):
        while self.running:
            try:
                msg, delay = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if delay > 0:
                    time.sleep(delay)
                self.engine.say(msg)
                self.engine.runAndWait()
            except Exception:
                pass

    def greet(self, name: str, delay: float = 0.0):
        self.q.put((f"Welcome to SymbiTech Twenty Twenty Five, {name}!", delay))

    def stop(self):
        self.running = False
        try: self.engine.stop()
        except: pass

# ==============================
# Font & Banner helpers
# ==============================
def _find_font_path():
    if FORCE_FONT_PATH and Path(FORCE_FONT_PATH).exists():
        return FORCE_FONT_PATH
    system = platform.system().lower()
    candidates = []
    if "windows" in system:
        candidates += [
            r"C:\Windows\Fonts\seguisb.ttf",
            r"C:\Windows\Fonts\segoeuib.ttf",
            r"C:\Windows\Fonts\segoeui.ttf",
            r"C:\Windows\Fonts\arialbd.ttf",
            r"C:\Windows\Fonts\arial.ttf",
        ]
    elif "darwin" in system or "mac" in system:
        candidates += [
            "/Library/Fonts/Arial Bold.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/HelveticaNeue.ttc",
        ]
    else:
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
    for p in candidates:
        if Path(p).exists():
            return p
    return None

def _load_font(px_size: int):
    fp = _find_font_path()
    if fp:
        try:
            return ImageFont.truetype(fp, px_size)
        except Exception:
            pass
    return ImageFont.load_default()

def _draw_text_with_outline(draw, pos, text, font, fill=(255,255,255), outline=(0,0,0), stroke=6):
    draw.text(pos, text, font=font, fill=fill, stroke_width=stroke, stroke_fill=outline)

def circle_crop(pil_img: Image.Image, size: int) -> Image.Image:
    pil_img = pil_img.copy().resize((size, size))
    mask = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(mask)
    d.ellipse((0, 0, size, size), fill=255)
    result = ImageOps.fit(pil_img, (size, size))
    result.putalpha(mask)
    return result

def _draw_hex_grid(img: Image.Image, spacing=48, color=(255,255,255,28)):
    w, h = img.size
    overlay = Image.new("RGBA", (w, h), (0,0,0,0))
    d = ImageDraw.Draw(overlay)
    a = spacing / 2
    b = spacing * np.sin(np.pi / 3)
    for row in range(int(h // b) + 2):
        y = int(row * b)
        x_offset = int((row % 2) * a)
        for col in range(int(w // spacing) + 2):
            x = int(col * spacing + x_offset)
            pts = [
                (x + a, y),(x + spacing, y + b/2),(x + spacing, y + 3*b/2),
                (x + a, y + 2*b),(x, y + 3*b/2),(x, y + b/2)
            ]
            d.line(pts + [pts[0]], fill=color, width=1)
    img.alpha_composite(overlay)
    return img

def _fit_font_to_box(text, start_px, max_w, max_h, stroke=6, min_px=18):
    px = start_px
    while px >= min_px:
        f = _load_font(px)
        tmp = Image.new("RGB", (1,1))
        bbox = ImageDraw.Draw(tmp).textbbox((0,0), text, font=f, stroke_width=stroke)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        if tw <= max_w and th <= max_h:
            return f
        px = int(px * 0.95)
    return _load_font(min_px)

def make_welcome_banner(width: int, height: int, name: str,
                        profile_path: str = "", linkedin: str = "", info_lines: List[str] = None) -> Image.Image:
    if info_lines is None:
        info_lines = []

    base = Image.new("RGBA", (width, height), (22, 26, 36, 255))
    draw = ImageDraw.Draw(base)
    top = np.array([22, 26, 36], dtype=np.float32)
    bottom = np.array([40, 80, 150], dtype=np.float32)
    for y in range(height):
        a = y / max(1, height - 1)
        c = ((1 - a) * top + a * bottom).astype(np.uint8)
        draw.line([(0, y), (width, y)], fill=tuple(c.tolist()), width=1)
    base = _draw_hex_grid(base, spacing=max(int(height/10), 40), color=(255,255,255,22))

    PAD = max(int(height * 0.04), 20)
    GUTTER = max(int(width * 0.02), 16)
    left_col = max(int(width * 0.32), 240)
    right_col_x = PAD + left_col + GUTTER
    right_col_w = max(width - right_col_x - PAD, 200)
    right_col_h = height - 2 * PAD
    right_col_y = PAD

    # Avatar
    avatar_size = min(int(height * 0.56), left_col - 2*PAD)
    avatar_x = PAD + int((left_col - avatar_size) / 2)
    avatar_y = PAD + int((right_col_h - avatar_size) / 2)
    avatar_card = Image.new("RGBA", (avatar_size + 44, avatar_size + 44), (255, 255, 255, 28))
    base.alpha_composite(avatar_card, (avatar_x - 22, avatar_y - 22))
    if profile_path and Path(profile_path).exists():
        try:
            pimg = Image.open(profile_path).convert("RGB")
            avatar = circle_crop(pimg, avatar_size)
            base.alpha_composite(avatar, (avatar_x, avatar_y))
        except Exception:
            pass

    # Text (right)
    title_line = WELCOME_LINE.format(name=name)
    sub_line   = "We’re glad you’re here."
    title_box_h = int(right_col_h * 0.60)
    sub_box_h   = int(right_col_h * 0.20)
    info_box_y  = right_col_y + title_box_h + sub_box_h
    info_box_h  = right_col_h - title_box_h - sub_box_h

    title_font = _fit_font_to_box(title_line, start_px=int(height * 0.18),
                                  max_w=right_col_w, max_h=title_box_h, stroke=6)
    sub_font   = _fit_font_to_box(sub_line,   start_px=int(height * 0.10),
                                  max_w=right_col_w, max_h=sub_box_h,   stroke=4)

    tmp = Image.new("RGB", (1,1))
    tbb = ImageDraw.Draw(tmp).textbbox((0,0), title_line, font=title_font, stroke_width=6)
    tw, th = tbb[2]-tbb[0], tbb[3]-tbb[1]
    tx = right_col_x + (right_col_w - tw)//2
    ty = right_col_y + (title_box_h - th)//2
    _draw_text_with_outline(draw, (tx, ty), title_line, title_font,
                            fill=(255,255,255), outline=(0,0,0), stroke=6)

    sbb = ImageDraw.Draw(tmp).textbbox((0,0), sub_line, font=sub_font, stroke_width=4)
    sw, sh = sbb[2]-sbb[0], sbb[3]-sbb[1]
    sx = right_col_x + (right_col_w - sw)//2
    sy = right_col_y + title_box_h + (sub_box_h - sh)//2
    _draw_text_with_outline(draw, (sx, sy), sub_line, sub_font,
                            fill=(235,235,235), outline=(0,0,0), stroke=4)

    info_y = info_box_y + int(info_box_h * 0.08)
    info_font = _load_font(max(int(height * 0.035), 18))
    for line in (info_lines or []):
        _draw_text_with_outline(draw, (right_col_x + 6, info_y), line, info_font,
                                fill=(230,230,230), outline=(0,0,0), stroke=3)
        info_y += int(info_font.size * 1.25)

    return base

# ==============================
# Main UI App (true 50/50 split)
# ==============================
class KioskApp:
    def __init__(self, root: tk.Tk, dataset_dir: Path):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.state("zoomed")
        self.root.configure(bg="#0f1115")
        self.root.bind("<Escape>", lambda e: self.on_close())

        # Half & half grid
        root.grid_rowconfigure(1, weight=1)
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=1)

        header = tk.Frame(root, bg="#0f1115")
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        tk.Label(header, text=APP_TITLE + " · Robotics & Automation",
                 fg="white", bg="#0f1115", font=("Segoe UI Semibold", 20)).pack(anchor="w", padx=20, pady=10)

        self.left = tk.Frame(root, bg="#0f1115")
        self.left.grid(row=1, column=0, sticky="nsew", padx=(20, 10), pady=(10, 20))
        self.left.grid_rowconfigure(0, weight=1)
        self.left.grid_columnconfigure(0, weight=1)
        self.video_label = tk.Label(self.left, bg="#121720")
        self.video_label.grid(row=0, column=0, sticky="nsew")

        self.right = tk.Frame(root, bg="#0f1115")
        self.right.grid(row=1, column=1, sticky="nsew", padx=(10, 20), pady=(10, 20))
        self.right.grid_rowconfigure(0, weight=1)
        self.right.grid_columnconfigure(0, weight=1)
        self.banner_holder = tk.Label(self.right, bg="#141821")
        self.banner_holder.grid(row=0, column=0, sticky="nsew")

        # Data
        self.dataset_dir = dataset_dir
        self.known_enc, self.known_names, self.profile_images = load_or_build_encodings(dataset_dir)
        self.guests = load_guests_info(Path(GUESTS_FILE))

        # Threads
        self.cam = CameraGrabber(CAM_INDEX, TARGET_CAM_W, TARGET_CAM_H); self.cam.start()
        self.recog = Recognizer(self.cam, self.known_enc, self.known_names); self.recog.start()
        self.audio = AudioGreeter(); self.audio.start()

        # State
        self.current_name = None
        self.showing_until = 0.0
        self.banner_tk = None

        # NEW: per-guest *speak-every* timer
        self.last_spoken_time: Dict[str, float] = {}  # name -> last time we enqueued speech

        self.banner_ready = False
        self.banner_holder.bind("<Configure>", self._on_banner_resize)

        # Loop
        self.update_loop()

    def _on_banner_resize(self, event):
        if event.width < 80 or event.height < 80:
            return
        self.banner_ready = True
        if self.current_name:
            self.render_welcome(self.current_name)
        else:
            self.render_idle()

    def draw_roi(self, frame):
        H, W = frame.shape[:2]
        x1, y1 = int(W * ROI_REL[0]), int(H * ROI_REL[1])
        x2, y2 = int(W * ROI_REL[2]), int(H * ROI_REL[3])

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 220, 140), thickness=-1)
        frame = cv2.addWeighted(overlay, 0.14, frame, 0.86, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 170), 3)
        cv2.putText(frame, "STEP INTO THE BOX", (x1 + 10, y1 - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 170), 3)
        return frame

    def update_loop(self):
        frame = self.cam.read()
        if frame is not None:
            annotated = self.draw_roi(frame.copy())
            boxes, names = self.recog.get_last()

            # Draw boxes/names
            for (top, right, bottom, left), name in zip(boxes, names):
                cv2.rectangle(annotated, (left, top), (right, bottom), (0, 255, 0), 2)
                (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
                cv2.rectangle(annotated, (left, bottom - 30), (left + tw + 12, bottom), (0, 255, 0), -1)
                cv2.putText(annotated, name, (left + 6, bottom - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2)

            # === SPEAK EACH TIME RECOGNIZED (with tiny spacing to avoid audio spam) ===
            recognized_names = [n for n in names if n != "Unknown"]
            now = time.time()
            for idx, nm in enumerate(recognized_names):
                last_t = self.last_spoken_time.get(nm, 0.0)
                if (now - last_t) >= SPEAK_EVERY_SECONDS:
                    self.last_spoken_time[nm] = now
                    # Update banner for the most recent spoken person
                    self.current_name = nm
                    self.showing_until = now + WELCOME_SECONDS
                    self.render_welcome(nm)
                    # Stagger slightly if multiple in the same frame
                    self.audio.greet(nm, delay=idx * 0.3)

            # Show camera
            disp = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            lw = self.video_label.winfo_width() or 960
            lh = self.video_label.winfo_height() or 540
            pil = Image.fromarray(disp); pil.thumbnail((lw, lh))
            self.video_tk = ImageTk.PhotoImage(pil)
            self.video_label.configure(image=self.video_tk)

            # Keep welcome for WELCOME_SECONDS, then idle
            if self.banner_ready:
                if (self.current_name is None) or (time.time() > self.showing_until):
                    self.current_name = None
                    self.render_idle()

        self.root.after(10, self.update_loop)

    def _render_banner(self, name: str, info_lines: List[str], linkedin: str, profile: str):
        bw = self.banner_holder.winfo_width()
        bh = self.banner_holder.winfo_height()
        if bw <= 80 or bh <= 80:
            self.root.after(80, lambda: self._render_banner(name, info_lines, linkedin, profile))
            return

        banner = make_welcome_banner(
            width=bw, height=bh,
            name=name,
            profile_path=profile,
            linkedin=linkedin,
            info_lines=info_lines
        )
        self.banner_tk = ImageTk.PhotoImage(banner)
        self.banner_holder.config(image=self.banner_tk)

    def render_idle(self):
        self._render_banner(
            name="Guest",
            info_lines=["Robotics & Automation", "SymbiTech 2025"],
            linkedin="",
            profile=""
        )

    def render_welcome(self, person_name: str):
        info = self.guests.get(person_name, {})
        profile = self.profile_images.get(person_name, "")
        lines = []
        if info.get("title"): lines.append(info["title"])
        if info.get("organization"): lines.append(info["organization"])
        self._render_banner(
            name=person_name,
            info_lines=lines,
            linkedin=info.get("linkedin",""),
            profile=profile
        )

    def on_close(self):
        try: self.recog.stop()
        except: pass
        try: self.cam.stop()
        except: pass
        try: self.audio.stop()
        except: pass
        self.root.destroy()

# ==============================
# Main
# ==============================
def main():
    try: cv2.setNumThreads(1)
    except: pass

    dataset_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dataset")
    if not dataset_dir.exists():
        print(f"[ERROR] Dataset not found: {dataset_dir}")
        sys.exit(1)

    root = tk.Tk()
    app = KioskApp(root, dataset_dir)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

if __name__ == "__main__":
    main()
