
import base64
import io

import numpy as np
import cv2
from PIL import Image as PILImage

def cv2_to_base64_string(cv2_img: np.ndarray) -> str:
    img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(img.astype("uint8"))
    rawBytes = io.BytesIO()
    pil_img.save(rawBytes, "PNG")
    rawBytes.seek(0)  # return to the start of the file
    b64_bytes = base64.b64encode(rawBytes.read())
    b64_string = b64_bytes.decode('utf-8')
    return b64_string