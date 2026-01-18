import hashlib
import io
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy
from PIL import Image

FRAME_IS_LANDSCAPE = True  # False if frame is oriented in portrait
FRAME_PIXELS_LONG = 800
FRAME_PIXELS_SHORT = 480

script_dir = Path(__file__).parent
ORIGINALS_DIR = script_dir / 'photos/originals'
PREPARED_DIR = script_dir / 'photos/prepared'
RECENCY_THRESHOLD = 60 * 60 * 24 * 7  # Seconds ago to consider pictures to be "recent" and prioritized
RECENCY_PRIORITY = 0.2  # Show a "recent" photo this percentage of time

FACE_DETECTION_NET_PATH = script_dir / 'face-detection.prototxt'
FACE_DETECTION_WEIGHTS_PATH = script_dir / 'res10_300x300_ssd_iter_140000_fp16.caffemodel'
FACE_DETECTION_CONFIDENCE_CUTOFF = 0.7  # [0-1] -> [0-100]% Confident

def crop(image: cv2.Mat) -> cv2.Mat:
    """Intelligently crop the image to fit display dimensions.

    Adapted from Waveshare's example code.
    """
    target_width = FRAME_PIXELS_LONG
    target_height = FRAME_PIXELS_SHORT
    if not FRAME_IS_LANDSCAPE:
        target_width = FRAME_PIXELS_SHORT
        target_height= FRAME_PIXELS_LONG

    img_height, img_width = image.shape[:2]
    img_aspect = img_width / img_height
    target_aspect = target_width / target_height

    # Calculate resize dimensions to maintain aspect ratio
    if img_aspect < target_aspect:
        resize = (target_width, int(target_width / img_aspect))
    else:
        resize = (int(target_height * img_aspect), target_height)

    image = cv2.resize(image, resize)
    img_height, img_width = image.shape[:2]

    # Calculate offset for centering
    x_off = int((img_width - target_width) / 2)
    y_off = int((img_height - target_height) / 2)
    assert x_off == 0 or y_off == 0, "Aspect ratio calculation error"

    try:
        shift_x, shift_y = target_faces(image, x_off, y_off)
    except Exception:
        shift_x, shift_y = target_saliency(image, x_off, y_off)

    x_off += shift_x
    y_off += shift_y

    # Perform final crop
    image = image[y_off:y_off + target_height, x_off:x_off + target_width]
    img_height, img_width = image.shape[:2]
    return image

def target_faces(image: cv2.Mat, x_off: int, y_off: int):
    # Use face detection for smart cropping
    img_height, img_width = image.shape[:2]

    # Feed image through DNN face-detection model
    model = cv2.dnn.readNetFromCaffe(FACE_DETECTION_NET_PATH, FACE_DETECTION_WEIGHTS_PATH)
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    model.setInput(blob)
    output = numpy.squeeze(model.forward())

    # Collect the high-confidence faces
    confident_faces = []
    for i in range(output.shape[0]):
        if output[i, 2] > FACE_DETECTION_CONFIDENCE_CUTOFF:
            face_box = (output[i, 3:7] * numpy.array([img_width, img_height, img_width, img_height])).astype(int)
            # print(f'Model Face Loc: {output[i, 3:7]}')
            # print(f'Image Face Loc: {face_box}')
            confident_faces.append(face_box)
    if len(confident_faces) == 0:
        raise RuntimError('No Faces Detected')

    # Compute the location of the weighted average (by area) of the confident faces
    faces_center_and_area = []
    area_sum = 0
    for face in confident_faces:
        center = ((face[2] - face[0]) / 2 + face[0], (face[3] - face[1]) / 2 + face[1])
        area = (face[2] - face[0]) * (face[3] - face[1])
        area_sum += area
        faces_center_and_area.append((center, area))

    crop_target = [0, 0]
    for face in faces_center_and_area:
        crop_target[0] += face[0][0] * face[1] / area_sum
        crop_target[1] += face[0][1] * face[1] / area_sum

    shift_x = shift_y = 0
    if x_off:
        img_center = int(img_width / 2)
        shift_x = max(min(int(crop_target[0]) - img_center, x_off), -x_off)
    else:
        img_center = int(img_height / 2)
        shift_y = max(min(int(crop_target[1]) - img_center, y_off), -y_off)

    # print(f'Face Target shift: {shift_x}, {shift_y}')
    return shift_x, shift_y

def target_saliency(image: cv2.Mat, x_off: int, y_off: int):
    # Use saliency detection for smart cropping
    img_height, img_width = image.shape[:2]

    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    (success, saliencyMap) = saliency.computeSaliency(image)
    saliencyMap = (saliencyMap * 255).astype("uint8")

    shift_x = shift_y = 0
    if y_off:  # Vertical cropping needed
        vert = numpy.max(saliencyMap, axis=1)
        vert = numpy.convolve(vert, numpy.ones(64)/64, "same")  # Smooth signal
        sal_center = int(numpy.argmax(vert))  # Most important vertical position
        img_center = int(img_height / 2)
        # Adjust crop to focus on salient region
        shift_y = max(min(sal_center - img_center, y_off), -y_off)
    else:  # Horizontal cropping needed
        horiz = numpy.max(saliencyMap, axis=0)
        horiz = numpy.convolve(horiz, numpy.ones(64)/64, "same")  # Smooth signal
        sal_center = int(numpy.argmax(horiz))  # Most important horizontal position
        img_center = int(img_width / 2)
        # Adjust crop to focus on salient region
        shift_x = max(min(sal_center - img_center, x_off), -x_off)

    return shift_x, shift_y

def display(image_path: Path):
    """Display prepared image on the e-paper device

    Adapted from Waveshare's example code.
    """
    import waveshare.epd7in3e as waveshare_driver

    try:
        epd = waveshare_driver.EPD()
        epd.init()
    except Exception as e:
        print(f"Failed to initialize display: {e}")
        return

    image = Image.open(image_path)
    if FRAME_IS_LANDSCAPE:
        image = image.rotate(180)  # Display seems to be upside down?
    else:
        image = image.rotate(90, expand=True)

    try:
        # Send image to display
        epd.display(epd.getbuffer(image))
        time.sleep(1)  # Wait for display to complete

        # Put display into low-power sleep mode
        epd.sleep()
    except AttributeError as e:
        print(f"Error: Driver module missing required attribute: {e}")
    except Exception as e:
        print(f"Error during display process")
        # Ensure proper cleanup
        if hasattr(epd, 'epdconfig'):
            epd.epdconfig.module_exit()
        raise e

def dither(image: Image) -> Image:
    """Dither for 6-color eInk display."""
    palette_image = Image.new("P", (1,1))
    palette_image.putpalette((0,0,0,  255,255,255,  255,0,0,  0,255,0,  0,0,255,  255,255,0))
    return image.quantize(colors=6, palette=palette_image).convert('RGB')

def prepare_image(original_path: Path, prepared_path: Path):
    image_cv2 = cv2.imread(original_path)
    image_cv2 = crop(image_cv2)
    _, buffer = cv2.imencode(".jpg", image_cv2)
    image_pil = Image.open(io.BytesIO(buffer))
    image_pil = dither(image_pil)
    image_pil.save(prepared_path)

def select_image():
    recency_cutoff = time.time() - RECENCY_THRESHOLD
    recent, old = [], []
    for candidate in ORIGINALS_DIR.glob('*.jpg', case_sensitive=False):
        if candidate.stat().st_ctime > recency_cutoff:
            recent.append(candidate)
        else:
            old.append(candidate)
    # print(f'lengths: recent: {len(recent)} -- old: {len(old)}')
    candidates = old
    if not old or (recent and random.random() < RECENCY_PRIORITY):
        # print('Selecting from recent')
        candidates = recent

    if not candidates:
        return None
    return random.choice(candidates)

if __name__ == '__main__':
    img_path = select_image()
    if not img_path:
        print('No image selected.  Quitting.')
        sys.exit(1)

    with open(img_path, 'rb') as img:
        hash = hashlib.file_digest(img, 'md5').hexdigest()
    prepared_path = PREPARED_DIR / f'{hash}_{'landscape' if FRAME_IS_LANDSCAPE else 'portrait'}.bmp'
    if not prepared_path.exists():
        prepare_image(img_path, prepared_path)
    display(prepared_path)
