import cv2
import tkinter as tk
from tkinter import filedialog
import ttkbootstrap as ttk
from PIL import Image, ImageTk
from deepface import DeepFace

# ----------------------------------
# GUI SETUP
# ----------------------------------

root = ttk.Window(themename="superhero")
root.title("Marco - Face Recognition")
root.geometry("800x650")

capture = None
referenceImg = None
match = False
running = False
counting = 0

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise RuntimeError("Failed to load OpenCV face detector")


# ----------------------------------
# LOAD REFERENCE PHOTO
# ----------------------------------

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
        status_label.config(text="Failed to load reference image")
        print("ERROR: Could not load reference image")
        return

    referenceImg = image
    match = False

    print("\nReference image loaded:", filepath)
    print("Reference image shape:", referenceImg.shape)

    status_label.config(text="Reference photo loaded")


# ----------------------------------
# VERIFY FACE
# ----------------------------------

def check_face(frame):
    global match

    if referenceImg is None:
        return

    try:
        print("\nChecking face...")

        result = DeepFace.verify(
            img1_path=frame,
            img2_path=referenceImg,
            model_name="SFace",
            detector_backend="opencv",
            enforce_detection=True,
            align=True
        )

        CUSTOM_THRESHOLD = 0.35

        distance = result["distance"]
        match = distance <= CUSTOM_THRESHOLD

        print("Distance:", distance)
        print("Custom threshold:", CUSTOM_THRESHOLD)
        print("Verified:", match)

        print("Verified:", result["verified"])
        print("Distance:", result["distance"])
        print("Threshold:", result["threshold"])

        if match:
            print("FACE MATCHED!")
        else:
            print("Face NOT matched.")

    except Exception as e:
        print("DeepFace ERROR:", repr(e))
        match = False


# ----------------------------------
# START CAMERA
# ----------------------------------

def start_capture():
    global capture, running, counting, match

    if referenceImg is None:
        status_label.config(text="Please select a reference photo first")
        return

    if running:
        return

    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not capture.isOpened():
        status_label.config(text="Could not open webcam")
        print("ERROR: Webcam not available")
        capture.release()
        capture = None
        return

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    running = True
    match = False
    counting = 0

    status_label.config(text="Camera running - searching for face")

    print("\nCamera started.")
    update_frame()


# ----------------------------------
# STOP CAMERA
# ----------------------------------

def stop_capture():
    global capture, running

    running = False

    if capture is not None:
        capture.release()
        capture = None

    print("Camera stopped.")


# ----------------------------------
# MATCH SUCCESS SCREEN
# ----------------------------------

def show_match_screen():

    # Switch off webcam
    stop_capture()

    # Remove all existing widgets
    for widget in root.winfo_children():
        widget.destroy()

    # Change background
    root.configure(bg="#111111")

    # Success message
    match_label = tk.Label(
        root,
        text="MATCH!",
        font=("Arial", 85, "bold"),
        fg="#00FF00",
        bg="#111111"
    )

    match_label.pack(expand=True)

    print("\nSUCCESS: Face recognised!")


# ----------------------------------
# UPDATE CAMERA FRAMES
# ----------------------------------

def update_frame():
    global counting, match

    if not running or capture is None:
        return

    ret, frame = capture.read()

    if not ret:
        print("ERROR: Failed to read webcam frame")
        stop_capture()
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(40, 40)
    )

    # Check recognition every 50 frames
    if len(faces) > 0 and counting % 10 == 0 and not match:
        check_face(frame)

    counting += 1

    # Successful match
    if match:
        show_match_screen()
        return

    # Draw face detection boxes
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

    # Display webcam frame in GUI
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=image)

    canvas.create_image(
        0,
        0,
        anchor=tk.NW,
        image=imgtk
    )

    canvas.imgtk = imgtk

    root.after(10, update_frame)


# ----------------------------------
# BUTTONS
# ----------------------------------

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

status_label = ttk.Label(
    root,
    text="Please select a reference photograph",
    font=("Arial", 12)
)
status_label.pack(pady=10)

canvas = ttk.Canvas(
    root,
    width=640,
    height=480
)
canvas.pack(pady=10)


# ----------------------------------
# CLEAN EXIT
# ----------------------------------

def on_close():
    stop_capture()
    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_close)

root.mainloop()