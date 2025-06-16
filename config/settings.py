# config/settings.py

from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = os.path.join(ROOT_DIR, 'data')
PICS_DIR = os.path.join(ROOT_DIR, 'pics')

PNG_PICS_DIR = os.path.join(PICS_DIR, 'png')
PDF_PICS_DIR = os.path.join(PICS_DIR, 'pdf')

PROPAGATION_DIR = os.path.join(DATA_DIR, 'propagation')

SPECTRUM_DIR = os.path.join(DATA_DIR, 'spectrum')

ELECTRONS_DIR = os.path.join(DATA_DIR, 'electrons')
SYNCH_TABLES_DIR = os.path.join(ELECTRONS_DIR, 'synch_tables')
SYNCH_INTERP_DIR = os.path.join(ELECTRONS_DIR, 'synch_interpolators')
JOINT_COOLING_DIR = os.path.join(ELECTRONS_DIR, 'joint_cooling_models')

INVERSE_COMPTON_DIR = os.path.join(DATA_DIR, 'inverse_compton')

PROTONS_DIR = os.path.join(DATA_DIR, 'protons')
ISRF_DIR = os.path.join(DATA_DIR, 'ISRF')
SHAPE_DIR = os.path.join(DATA_DIR, 'shape')
MCMC_DIR = os.path.join(DATA_DIR, 'mcmc_samples')

MCMC_ELECTRONS_SYNCH_ONLY = os.path.join(MCMC_DIR, 'electrons_synch_only')
MCMC_ELECTRONS_JOINT = os.path.join(MCMC_DIR, 'electrons_joint')
MCMC_ELECTRONS_STEADY_JOINT = os.path.join(MCMC_DIR, 'electrons_joint_steady')
MCMC_ELECTRONS_SYNCH_STEADY = os.path.join(MCMC_DIR, 'electrons_synch_steady')

MCMC_PROTONS = os.path.join(MCMC_DIR, 'protons')
