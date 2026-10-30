
import cv2
import numpy as np
import tkinter as tk
import ttkbootstrap as ttk
import pyrealsense2 as rs

from tkinter import filedialog
from PIL import Image, ImageTk
from deepface import DeepFace

WIDTH = 640
HEIGHT = 480
FPS = 15

CUSTOM_THRESHOLD = 0.35
CHECK_INTERVAL = 10

root = ttk.Window(themename="superhero")
root.title("Marco - Face Recognition")
root.geometry("800x700")

pipeline = None
referenceImg = None
match = False
running = False
counting = 0
active_mode = None

camera_mode = tk.StringVar(value="RGB")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise RuntimeError("Failed to load OpenCV face detector")

def load_reference():
    global referenceImg, match

    filepath = filedialog.askopenfilename(
        title="Select Reference Photo",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png"),
            ("All Files", "*.*")
        ]
    )

    if not filepath:
        return

    image = cv2.imread(filepath)

    if image is None:
        status_label.config(
            text="Failed to load reference image"
        )
        return

    referenceImg = image
    match = False

    print("\nReference image loaded:", filepath)
    print("Image shape:", referenceImg.shape)

    status_label.config(text="Reference photo loaded")


def check_face(face_image):
    global match

    if referenceImg is None:
        return

    try:
        print("\nChecking face...")

        result = DeepFace.verify(
            img1_path=face_image,
            img2_path=referenceImg,
            model_name="SFace",
            detector_backend="opencv",
            enforce_detection=True,
            align=True
        )

        distance = result["distance"]
        match = distance <= CUSTOM_THRESHOLD

        print("Distance:", distance)
        print("Custom threshold:", CUSTOM_THRESHOLD)
        print("DeepFace threshold:", result["threshold"])
        print("Verified by DeepFace:", result["verified"])
        print("Custom match:", match)

        if match:
            print("FACE MATCHED!")
        else:
            print("Face NOT matched.")

    except Exception as e:
        print("DeepFace ERROR:", repr(e))
        match = False


def start_capture():
    global pipeline, running, counting, match, active_mode

    if referenceImg is None:
        status_label.config(
            text="Please select a reference photo first"
        )
        return

    if running:
        return

    active_mode = camera_mode.get()

    pipeline = rs.pipeline()
    config = rs.config()

    # RGB camera
    if active_mode == "RGB":
        config.enable_stream(
            rs.stream.color,
            WIDTH,
            HEIGHT,
            rs.format.bgr8,
            FPS
        )

    # Infrared camera
    else:
        config.enable_stream(
            rs.stream.infrared,
            1,
            WIDTH,
            HEIGHT,
            rs.format.y8,
            FPS
        )

    try:
        profile = pipeline.start(config)

        # Enable infrared projector if available.
        # This is useful for low-light IR viewing.
        if active_mode == "Infrared":
            device = profile.get_device()

            for sensor in device.query_sensors():
                if sensor.supports(rs.option.emitter_enabled):
                    sensor.set_option(
                        rs.option.emitter_enabled, 1
                    )
                    break

    except Exception as e:
        print("RealSense ERROR:", repr(e))

        if pipeline is not None:
            try:
                pipeline.stop()
            except RuntimeError:
                pass

        pipeline = None

        status_label.config(
            text="Could not start RealSense camera"
        )
        return

    running = True
    match = False
    counting = 0

    print("\nRealSense D415 started")
    print("Mode:", active_mode)

    status_label.config(
        text=f"{active_mode} camera running - searching for face"
    )

    update_frame()

def stop_capture():
    global pipeline, running

    running = False

    if pipeline is not None:
        try:
            pipeline.stop()
        except RuntimeError as e:
            print("Camera stop error:", e)

        pipeline = None

    print("Camera stopped.")


def show_match_screen():

    stop_capture()

    for widget in root.winfo_children():
        widget.destroy()

    root.configure(bg="#111111")

    match_label = tk.Label(
        root,
        text="MATCH!",
        font=("Arial", 85, "bold"),
        fg="#00FF00",
        bg="#111111"
    )

    match_label.pack(expand=True)

    print("\nSUCCESS: Face recognised!")


def update_frame():
    global counting, match

    if not running or pipeline is None:
        return

    try:
        # Non-blocking frame retrieval
        frames = pipeline.poll_for_frames()

        if not frames:
            root.after(10, update_frame)
            return

        # Read RGB or infrared frame
        if active_mode == "RGB":
            camera_frame = frames.get_color_frame()
        else:
            camera_frame = frames.get_infrared_frame(1)

        if not camera_frame:
            root.after(10, update_frame)
            return

        frame = np.asanyarray(
            camera_frame.get_data()
        )

        # Infrared is grayscale.
        # Convert to BGR for OpenCV/DeepFace.
        if active_mode == "Infrared":
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_GRAY2BGR
            )

        # ----------------------------------
        # FACE DETECTION
        # ----------------------------------

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(40, 40)
        )



        if (
            len(faces) > 0
            and counting % CHECK_INTERVAL == 0
            and not match
        ):
            # Compare each detected face separately.
            # Include some surrounding image so DeepFace
            # can detect and align the face.
            for x, y, w, h in faces:
                padding = int(max(w, h) * 0.25)

                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(frame.shape[1], x + w + padding)
                y2 = min(frame.shape[0], y + h + padding)

                face_crop = frame[y1:y2, x1:x2].copy()

                check_face(face_crop)

                if match:
                    break

        counting += 1

        if match:
            show_match_screen()
            return


        for x, y, w, h in faces:
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                2
            )

        cv2.putText(
            frame,
            "NOT MATCHED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        # Show current camera mode
        cv2.putText(
            frame,
            active_mode,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

  

        image = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(image)
        imgtk = ImageTk.PhotoImage(image=image)

        canvas.create_image(
            0,
            0,
            anchor=tk.NW,
            image=imgtk
        )

        canvas.imgtk = imgtk

    except Exception as e:
        print("Frame ERROR:", repr(e))
        stop_capture()
        status_label.config(text="Camera frame error")
        return

    root.after(10, update_frame)



title_label = ttk.Label(
    root,
    text="MARCO - Face Recognition",
    font=("Arial", 20, "bold")
)
title_label.pack(pady=15)


# Camera selection
mode_frame = ttk.Frame(root)
mode_frame.pack(pady=5)

ttk.Label(
    mode_frame,
    text="Camera mode:",
    font=("Arial", 12)
).pack(side=tk.LEFT, padx=10)

mode_selector = ttk.Combobox(
    mode_frame,
    textvariable=camera_mode,
    values=["RGB", "Infrared"],
    state="readonly",
    width=15
)
mode_selector.pack(side=tk.LEFT, padx=10)


# Buttons
button_frame = ttk.Frame(root)
button_frame.pack(pady=15)

upload_button = ttk.Button(
    button_frame,
    text="Select Reference Photo",
    command=load_reference,
    bootstyle="info-outline"
)
upload_button.pack(side=tk.LEFT, padx=10)

start_button = ttk.Button(
    button_frame,
    text="Start Capture",
    command=start_capture,
    bootstyle="success-outline"
)
start_button.pack(side=tk.LEFT, padx=10)

stop_button = ttk.Button(
    button_frame,
    text="Stop Capture",
    command=stop_capture,
    bootstyle="danger-outline"
)
stop_button.pack(side=tk.LEFT, padx=10)


# Status label
status_label = ttk.Label(
    root,
    text="Please select a reference photograph",
    font=("Arial", 12)
)
status_label.pack(pady=10)


# Camera display
canvas = ttk.Canvas(
    root,
    width=WIDTH,
    height=HEIGHT
)
canvas.pack(pady=10)



def on_close():
    stop_capture()
    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_close)

root.mainloop()
