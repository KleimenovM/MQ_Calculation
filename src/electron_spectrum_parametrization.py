import numpy as np


class ElectronSpectrumParametrization:
    def __init__(self, n0, e0, eta0, p0, k10, k20):
        """
        Set the four-parameter spectrum parametrization class
        :param n0: [eV-1 cm-3], reference spectral energy density
        :param e0: [eV], reference energy
        :param eta0: [DL], normalization factor
        :param p0: [DL] spectral index
        :param k10: [DL], low energy limit factor
        :param k20: [DL], upper energy limit factor
        """
        self.n0 = n0
        self.e0 = e0
        self.eta0 = eta0
        self.p0 = p0
        self.k10 = k10
        self.k20 = k20

    def dn_de(self, e, eta, p, k1, k2):
        """
        Electron density parametrization
        :param e: [eV], energy
        :param eta: [DL], normalization factor
        :param p: [DL], power law index
        :param k1: [DL], low energy limit factor
        :param k2: [DL], upper energy limit factor
        :return: [eV-1 cm-3] spectral curve (dn/(de dV))
        """
        e_min = 10**k1 * self.e0
        e_max = 10**k2 * self.e0
        return self.n0 * np.exp(eta - p * np.log(e / self.e0) - e / e_max) * np.heaviside(e - e_min, 1.0)

    def dn_de0(self, e):
        """
        Electron density parametrization
        :param e:
        :return:
        """
        return self.dn_de(e, self.eta0, self.p0, self.k10, self.k20)

    def tot_n(self, eta, p, k1, k2):
        """
        Total electron number density
        :param eta:
        :param p:
        :param k1:
        :param k2:
        :return:
        """
        e = self.e0 * 10**np.linspace(k1, k2, 1000)
        dn_de = self.dn_de(e, eta, p, k1, k2)
        return np.trapezoid(dn_de, e)

    def tot_n0(self):
        """
        Total electron number density
        :return:
        """
        return self.tot_n(self.eta0, self.p0, self.k10, self.k20)

    def tot_e(self, eta, p, k1, k2):
        """
        Total electron energy density
        :param eta:
        :param p:
        :param k1:
        :param k2:
        :return:
        """
        e = self.e0 * 10 ** np.linspace(k1, k2, 1000)
        dn_de = self.dn_de(e, eta, p, k1, k2)
        return np.trapezoid(e * dn_de, e)

    def tot_e0(self):
        """
        Total electron energy density
        :return:
        """
        return self.tot_e(self.eta0, self.p0, self.k10, self.k20)


if __name__ == '__main__':
    print("Not for direct use")
