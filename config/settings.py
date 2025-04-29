# config/settings.py

from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = os.path.join(ROOT_DIR, 'data')
PICS_DIR = os.path.join(ROOT_DIR, 'pics')

SPECTRUM_DIR = os.path.join(DATA_DIR, 'spectrum')
ELECTRONS_DIR = os.path.join(DATA_DIR, 'electrons')
PROTONS_DIR = os.path.join(DATA_DIR, 'protons')
ISRF_DIR = os.path.join(DATA_DIR, 'isrf')
