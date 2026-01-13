# ePhotoFrame

A set of scripts to display photos on a [Waveshare ePaper display (7.5 inch, Full color)](https://www.waveshare.com/rpi-zero-photopainter-acce.htm?sku=33398) Powered by a Raspberry Pi Zero 2 W.

From the available photos a random photo will be selected and displayed.

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
7. Preprocess photos into dithered BMPs using the `waveshare/convert-all.py` script.
    - `cd .../location/of/JPGs && ./venv/bin/python waveshare/convert-all.py`
8. Copy generated BMPs to `.../ePhotoFrame/photos`.
9. Run application `.venv/bin/python main.py`

# Cron

Example Crontab entries using Flock to avoid simultaneous executions (which usually result in image corruption on the display):

    > # Update every 15 minutes during the day
    > */15 * * * * /usr/bin/flock -n /home/pi/ePhotoFrame_cron.lock -c "/home/pi/ePhotoFrame/.venv/bin/python /home/pi/ePhotoFrame/main.py 2>&1 | /usr/bin/logger -t ePhotoFrame"

# TODO:

1. Handle preprocessing files into dithered BMPs at runtime if needed.

