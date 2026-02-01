# ePhotoFrame

A set of scripts to display photos on a [Waveshare ePaper display (7.5 inch, Full color)](https://www.waveshare.com/rpi-zero-photopainter-acce.htm?sku=33398) Powered by a Raspberry Pi Zero 2 W.

Place photos in `photos/originals`.  The script will randomly select a photo from that directory and display it.  The selected image is automatically resized and cropped to fill the display.  Cropping is done by performing face detection and targeting on the location of the weighted average of detected faces.  If no faces are detected, the crop will target the location of highest "saliency" (See https://docs.opencv.org/3.4/df/d37/classcv_1_1saliency_1_1StaticSaliencySpectralResidual.html#details).  The cropped image is dithered for the 6-color display and the prepared image is saved, by hash, in `photos/prepared` so preparation is only performed once per image.

Image select is biased towards new images in the directory using `RECENCY_THRESHOLD`, to control what is considered "new," and `RECENCY_PRIORITY`, to control the bias.  The default behavior is to consider files added within the past week as "new" and to select a "new" image 20% of the time.

For automation of adding images, I use a cronjob with `nextcloudcmd` to automatically sync pictures from my NextCloud instance to `photos/originals`.

Written for Python 3.13

## Setup

1. After installing the RPi image, use the RPi config gui or `raspi-config` command to enable SPI (under "Interfaces").
2. Install python3.13 and libpython3.13-dev: `sudo apt install python3.13 libpython3.13-dev`.
3. Install gpiozero: `sudo apt install python3-gpiozero`.
4. Create virtual environment: `cd ePhotoFrame && python3 -m venv --system-site-packages .venv`
    - We use `--system-site-packages` so we can get the system `gpiozero` installation.
    - As of this writing, installing `gpiozero` in the venv doesn't work.
5. Install/upgrade pip: `.venv/bin/python -m pip install --upgrade pip`.
6. Install dependencies: `.venv/bin/python -m pip install -r rpi-requirements.txt`
7. Copy pictures into `photos/originals`.
8. Run application `.venv/bin/python main.py`

# Cron

Example Crontab entries using Flock to avoid simultaneous executions (which usually result in image corruption on the display):

    > # Update every 15 minutes during the day
    > */15 * * * * /usr/bin/flock -n /home/pi/ePhotoFrame_cron.lock -c "/home/pi/ePhotoFrame/.venv/bin/python /home/pi/ePhotoFrame/main.py 2>&1 | /usr/bin/logger -t ePhotoFrame"
