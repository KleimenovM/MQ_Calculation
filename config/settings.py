# config/settings.py

from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = os.path.join(ROOT_DIR, 'data')
PICS_DIR = os.path.join(ROOT_DIR, 'pics')

SPECTRUM_DIR = os.path.join(DATA_DIR, 'spectrum')
ELECTRONS_DIR = os.path.join(DATA_DIR, 'electrons')
PROTONS_DIR = os.path.join(DATA_DIR, 'protons')
ISRF_DIR = os.path.join(DATA_DIR, 'ISRF')
SHAPE_DIR = os.path.join(DATA_DIR, 'shape')
MCMC_DIR = os.path.join(DATA_DIR, 'mcmc_samples')

MCMC_ELECTRONS_SYNCH_ONLY = os.path.join(MCMC_DIR, 'electrons_synch_only')
MCMC_PROTONS = os.path.join(MCMC_DIR, 'protons')
