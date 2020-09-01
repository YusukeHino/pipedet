
import os
import random
import json
import copy
from enum import IntEnum, unique
from typing import List, Tuple, Optional, Union, Any

import requests
import numpy as np
import cv2

from .large_image import Image, LargeImage


class Frame(LargeImage):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)