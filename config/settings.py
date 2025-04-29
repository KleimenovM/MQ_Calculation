# config/settings.py

from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = os.path.join(ROOT_DIR, 'data')
PICS_DIR = os.path.join(ROOT_DIR, 'pics')
