import random
from pathlib import Path

from display_picture import run as show_picture

if __name__ == '__main__':
    script_dir = Path(__file__).parent
    candidates = list(script_dir.glob('photos/*.bmp'))
    img = random.choice(candidates)
    show_picture(img)

# TODO: Don't require pictures were already dithered into bmps.
# Load photos from configured directory, dither and store in another location
# Note: non-preprocessed images are ending up upside down.
