#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
#                                                                             #
# RMG - Reaction Mechanism Generator                                          #
#                                                                             #
# Copyright (c) 2002-2018 Prof. William H. Green (whgreen@mit.edu),           #
# Prof. Richard H. West (r.west@neu.edu) and the RMG Team (rmg_dev@mit.edu)   #
#                                                                             #
# Permission is hereby granted, free of charge, to any person obtaining a     #
# copy of this software and associated documentation files (the 'Software'),  #
# to deal in the Software without restriction, including without limitation   #
# the rights to use, copy, modify, merge, publish, distribute, sublicense,    #
# and/or sell copies of the Software, and to permit persons to whom the       #
# Software is furnished to do so, subject to the following conditions:        #
#                                                                             #
# The above copyright notice and this permission notice shall be included in  #
# all copies or substantial portions of the Software.                         #
#                                                                             #
# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,    #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER      #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING     #
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER         #
# DEALINGS IN THE SOFTWARE.                                                   #
#                                                                             #
###############################################################################

import os
import numpy as np

from rmgpy.thermo import ThermoData
from dde.predictor import Predictor


class MLEstimator():

    """
    A machine learning based estimator for thermochemistry prediction.

    The attributes are:

    =================== ======================= ====================================
    Attribute           Type                    Description
    =================== ======================= ====================================
    `Hf298_estimator`   :class:`Predictor`      Hf298 Estimator 
    `S298_estimator`    :class:`Predictor`      S298 Estimator 
    `Cp_estimator`      :class:`Predictor`      Cp Estimator 
    =================== ======================= ====================================

    """

    def __init__(self, Hf298_path, S298_path, Cp_path):

        self.Hf298_estimator = load_pretrained_estimator(Hf298_path)
        self.S298_estimator = load_pretrained_estimator(S298_path)
        self.Cp_estimator = load_pretrained_estimator(Cp_path)

    def get_thermo_data(self, molecule):
        """
        Return thermodynamic parameters corresponding to a given
        :class:`Molecule` object `molecule`.

        Returns: ThermoData
        """
        Tdata = [300.0, 400.0, 500.0, 600.0, 800.0, 1000.0, 1500.0]
        Cp = self.Cp_estimator.predict(molecule=molecule)
        Cp = [np.float64(Cp_i) for Cp_i in Cp]
        Hf298 = self.Hf298_estimator.predict(molecule=molecule)
        S298 = self.S298_estimator.predict(molecule=molecule)
        comment = "ML Estimation."

        Cp0 = molecule.calculateCp0()
        CpInf = molecule.calculateCpInf()
        thermo = ThermoData(
            Tdata=(Tdata, "K"),
            Cpdata=(Cp, "cal/(mol*K)"),
            H298=(Hf298, "kcal/mol"),
            S298=(S298, "cal/(mol*K)"),
            Cp0=(Cp0, "J/(mol*K)"),
            CpInf=(CpInf, "J/(mol*K)"),
            Tmin=(300.0, "K"),
            Tmax=(2000.0, "K"),
            comment=comment
        )
        return thermo

    def get_thermo_data_for_species(self, species):
        """
        Return the set of thermodynamic parameters corresponding to a given
        :class:`Species` object `species`.

        The current ML estimator treats each resonance isomer identically,
        i.e., any of the resonance isomers can be chosen.

        Returns: ThermoData
        """
        return self.get_thermo_data(species.molecule[0])


def load_pretrained_estimator(model_path):
    estimator = Predictor()
    predictor_input = os.path.join(model_path, 'predictor_input.py')
    param_path = os.path.join(model_path, 'full_train.h5')
    estimator.load_input(predictor_input)
    estimator.load_parameters(param_path=param_path)
    return estimator
