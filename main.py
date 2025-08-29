import cv2
import numpy as np
from nicegui import ui
import base64
from io import BytesIO
import requests
import matplotlib.pyplot as plt

# Global variables
current_image = None
cap = None
timer = None
camera_opened = False  # เพิ่มตัวแปรสถานะ

# Image processing parameters
brightness = 0
contrast = 1.0
gaussian_blur = 0
gamma = 1.0
sat_r = 1.0
sat_g = 1.0
sat_b = 1.0
threshold_value = 127
edge_mode = "none"


def cv_to_base64(img, quality=95):
    """Convert OpenCV image to base64 Data URL (supports JPG/PNG)"""
    if len(img.shape) == 2:  # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ext = ".jpg" if quality < 100 else ".png"
    encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext == ".jpg" else []
    _, buffer = cv2.imencode(ext, img, encode_param)
    mime = "jpeg" if ext == ".jpg" else "png"
    return f"data:image/{mime};base64," + base64.b64encode(buffer).decode("utf-8")


def plot_histogram(img):
    """Create histogram of image as base64"""
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    plt.figure(figsize=(5, 3))
    colors = ("b", "g", "r")
    for i, col in enumerate(colors):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
    plt.xlim([0, 256])
    plt.title("RGB Histogram")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    plt.close()
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")


def apply_adjustments(img):
    """Apply Brightness, Contrast, Saturation, Gamma, Blur"""
    img = img.astype(np.float32)
    img[:, :, 0] *= sat_b
    img[:, :, 1] *= sat_g
    img[:, :, 2] *= sat_r
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
    if gaussian_blur > 0:
        k = 2 * int(gaussian_blur) + 1
        img = cv2.GaussianBlur(img, (k, k), 0)
    if gamma != 1.0:
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(
            "uint8"
        )
        img = cv2.LUT(img, table)
    return img


def apply_edge(img):
    """Apply edge detection or image transformation based on mode"""
    if edge_mode == "none":
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    if edge_mode == "canny":
        return cv2.Canny(gray, 50, 150)
    elif edge_mode == "binary":
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        return binary
    elif edge_mode == "dilate":
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        return cv2.dilate(binary, kernel, iterations=1)
    elif edge_mode == "erode":
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        return cv2.erode(binary, kernel, iterations=1)
    elif edge_mode == "sobel":
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        return np.uint8(np.absolute(cv2.magnitude(sobelx, sobely)))
    elif edge_mode == "laplacian":
        return np.uint8(np.absolute(cv2.Laplacian(gray, cv2.CV_64F)))
    elif edge_mode == "gaussian":
        return cv2.GaussianBlur(gray, (5, 5), 0)
    elif edge_mode == "grayscale":
        return gray
    return img


def update_process():
    """Update processed image and histogram"""
    if current_image is None:
        return
    try:
        img = apply_adjustments(current_image)
        img = apply_edge(img)
        processed_ui.set_source(cv_to_base64(img, quality=95))
        hist_ui.set_source(plot_histogram(img))
    except Exception as e:
        ui.notify(f"Processing error: {str(e)}", type="error")


# === Input Handlers ===
def update_frame():
    """Update live camera frame"""
    ret, frame = cap.read()
    if ret:
        global current_image
        current_image = frame.copy()
        img_base64 = cv_to_base64(frame, quality=70)  # Lower quality for performance
        live_ui.set_source(img_base64)
        update_process()  # Update processed view and histogram
    # else: silently ignore dropped frames


def open_camera():
    global cap, timer, current_image, camera_opened
    if timer:
        timer.deactivate()
    if cap:
        cap.release()
    cap = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        ui.notify(
            "Failed to open camera. Check connection or permissions.", type="error"
        )
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    ret, _ = cap.read()
    if not ret:
        ui.notify("Camera opened but cannot read frames.", type="error")
        cap.release()
        cap = None
        return

    timer = ui.timer(0.05, update_frame)  # ~20 FPS
    ui.notify("Camera opened successfully", type="positive")
    camera_opened = True

    control_buttons.clear()
    with control_buttons:

        ui.button("Capture Photo", on_click=capture_image).props("color=green")
        ui.button("Close Camera", on_click=close_camera).props("color=red")


def capture_image():
    global current_image, timer
    if current_image is not None:
        if timer:
            timer.deactivate()
        original_ui.set_source(cv_to_base64(current_image, quality=95))
        update_process()
        ui.notify("Photo captured successfully", type="positive")

    # แก้: clear เฉพาะปุ่ม
    control_buttons.clear()
    with control_buttons:
        ui.button("Re-Capture", on_click=open_camera).props("color=blue")
        ui.button("Close Camera", on_click=close_camera).props("color=red")


def close_camera():
    global cap, timer, camera_opened
    if timer:
        timer.deactivate()
    if cap:
        cap.release()
    cap = None
    timer = None
    live_ui.set_source("")  # Clear live feed
    ui.notify("Camera closed", type="info")
    camera_opened = False  # กล้องปิดแล้ว

    camera_controls.clear()
    with camera_controls:
        ui.button("Open Camera", on_click=open_camera).props("color=blue")


def load_url():
    try:
        response = requests.get(url_input.value.strip(), timeout=5)
        response.raise_for_status()
        img_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is not None:
            global current_image
            current_image = img
            original_ui.set_source(cv_to_base64(img, quality=95))
            update_process()
            ui.notify("Image loaded from URL", type="positive")
        else:
            ui.notify("Failed to decode image from URL", type="error")
    except Exception as e:
        ui.notify(f"Error loading URL: {str(e)}", type="error")


def upload_image(e):
    global current_image
    content = e.content.read()
    img_array = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is not None:
        current_image = img
        original_ui.set_source(cv_to_base64(img, quality=95))
        update_process()
        ui.notify("Image uploaded successfully", type="positive")
    else:
        ui.notify("Failed to read uploaded image", type="error")


def set_edge_mode(mode):
    global edge_mode
    edge_mode = mode
    update_process()


# === Slider Handlers ===
def set_brightness(e):
    global brightness
    brightness = e.value
    update_process()


def set_contrast(e):
    global contrast
    contrast = e.value
    update_process()


def set_gaussian_blur(e):
    global gaussian_blur
    gaussian_blur = e.value
    update_process()


def set_gamma(e):
    global gamma
    gamma = e.value
    update_process()


def set_sat_r(e):
    global sat_r
    sat_r = e.value
    update_process()


def set_sat_g(e):
    global sat_g
    sat_g = e.value
    update_process()


def set_sat_b(e):
    global sat_b
    sat_b = e.value
    update_process()


def set_threshold(e):
    global threshold_value
    threshold_value = e.value
    update_process()


# === UI Layout ===
ui.label("Dashboard Image Processing").classes("text-h4")

# Create UI elements
with ui.row().classes("w-full gap-8"):
    with ui.column().classes("gap-2"):
        ui.label("Input Image").classes("text-lg font-bold")

        camera_controls = ui.column()
        with camera_controls:
            ui.button("Open Camera", on_click=open_camera).props("color=blue")

        control_buttons = ui.column()

        with ui.row():
            url_input = ui.input("Image URL").props('placeholder="https://"')
            ui.button("Load", on_click=load_url)

        ui.upload(on_upload=upload_image).props(
            'label="Drag & Drop Image" accept="image/*"'
        )

        ui.label("Adjust Parameters").classes("text-lg font-bold")
        ui.label("Brightness")
        ui.slider(min=-100, max=100, value=0, step=1).on_value_change(set_brightness)
        ui.label("Contrast")
        ui.slider(min=0.1, max=3.0, value=1.0, step=0.1).on_value_change(set_contrast)
        ui.label("Gaussian Blur")
        ui.slider(min=0, max=10, value=0, step=0.5).on_value_change(set_gaussian_blur)
        ui.label("Gamma")
        ui.slider(min=0.1, max=5.0, value=1.0, step=0.1).on_value_change(set_gamma)
        ui.label("Saturation R")
        ui.slider(min=0.0, max=2.0, value=1.0, step=0.1).on_value_change(set_sat_r)
        ui.label("Saturation G")
        ui.slider(min=0.0, max=2.0, value=1.0, step=0.1).on_value_change(set_sat_g)
        ui.label("Saturation B")
        ui.slider(min=0.0, max=2.0, value=1.0, step=0.1).on_value_change(set_sat_b)
        ui.label("Threshold Value")
        ui.slider(min=0, max=255, value=127, step=1).on_value_change(set_threshold)

        ui.label("Edge Detection").classes("text-lg font-bold")
        with ui.row(wrap=True):
            ui.button("Canny", on_click=lambda: set_edge_mode("canny")).props("outline")
            ui.button("Binary", on_click=lambda: set_edge_mode("binary")).props(
                "outline"
            )
            ui.button("Dilate", on_click=lambda: set_edge_mode("dilate")).props(
                "outline"
            )
            ui.button("Erode", on_click=lambda: set_edge_mode("erode")).props("outline")
            ui.button("Sobel", on_click=lambda: set_edge_mode("sobel")).props("outline")
            ui.button("Laplacian", on_click=lambda: set_edge_mode("laplacian")).props(
                "outline"
            )
            ui.button("Gaussian", on_click=lambda: set_edge_mode("gaussian")).props(
                "outline"
            )
            ui.button("Grayscale", on_click=lambda: set_edge_mode("grayscale")).props(
                "outline"
            )
            ui.button("Reset", on_click=lambda: set_edge_mode("none")).props("flat")

    with ui.column().classes("gap-2"):
        ui.label("Live Camera").classes("text-lg font-bold")
        live_ui = ui.image().style(
            "width: 400px; height: 300px; border: 1px solid #ccc; background: #000;"
        )

        ui.label("Original Image").classes("text-lg font-bold")
        original_ui = ui.image().style(
            "width: 400px; height: 300px; border: 1px solid #ddd;"
        )

        ui.label("Processed Image").classes("text-lg font-bold")
        processed_ui = ui.image().style(
            "width: 400px; height: 300px; border: 2px solid #000;"
        )

ui.label("Histogram RGB").classes("text-lg font-bold")
hist_ui = ui.image().style("width: 400px; height: 300px; background: #f8f8f8;")

# Start the server
ui.run(port=8080, title="Image Processor", favicon="🖼️")
