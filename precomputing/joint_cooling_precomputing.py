import os
import pickle
import time

import numpy as np

from precomputing.transport_equation_solution import energy_backpropagation, get_the_modulation_coefficient
from config.settings import JOINT_COOLING_DIR
from config.units import Gauss
from src.electron_cooling import JointCooling


def joint_cooling_precomputing(bvalue):
    """
    Performs energy backpropagation and modulation coefficient calculation for a given magnetic field.
    :param bvalue: Magnetic field value in Gauss
    :return: times, energies, modulation_coefficient
    """
    jc = JointCooling(bvalue)  # joing cooling set
    t0 = time.time()
    print("energy backpropagation calculation started")
    times, energies = energy_backpropagation(cooling=jc, bvalue=bvalue)
    t1 = time.time()
    print(f"energy backpropagation calculation finished in {t1 - t0:.0f} s")

    # modulation coefficient calculation
    t2 = time.time()
    print("modulation coefficient calculation started")
    modulation_coefficient = get_the_modulation_coefficient(cooling=jc, time_sol=times, energy_sol=energies)
    t3 = time.time()
    print(f"modulation coefficient calculation finished in {t3 - t2:.0f} s")
    return times, energies, modulation_coefficient


def save_cooling_precomputed(bvalue, times, energies, modulation_coefficient):
    with open(os.path.join(JOINT_COOLING_DIR, f"joint_cooling_{bvalue.value * 1e6:.1f}.pck"), "wb") as write_file:
        pickle.dump([times, energies, modulation_coefficient], write_file)
    return


if __name__ == '__main__':
    bfs = np.array([5]) * 1e-6 * Gauss
    for bf in bfs:
        print(f"MF = {bf.value * 1e6:.1f} uG")
        t, e, mc = joint_cooling_precomputing(bf)
        save_cooling_precomputed(bf, t, e, mc)
