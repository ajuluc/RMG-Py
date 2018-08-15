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
import unittest

from rmgpy import ml
from rmgpy.molecule import Molecule
from rmgpy.ml.estimator import MLEstimator


class TestMLEstimator(unittest.TestCase):

    def setUp(self):
        pretrained_models_path = os.path.join(ml.__path__[0], 'pretrained_models')
        Hf298_path = os.path.join(pretrained_models_path, 'Hf298')
        S298_path = os.path.join(pretrained_models_path, 'S298')
        Cp_path = os.path.join(pretrained_models_path, 'Cp')
        self.ml_estimator = MLEstimator(Hf298_path, S298_path, Cp_path)

    def test_get_thermo_data(self):
        mol = Molecule().fromSMILES('C1C2C1C2')
        thermo = self.ml_estimator.get_thermo_data(mol)

        self.assertTrue(thermo.comment.startswith('ML Estimation.'))
        self.assertAlmostEqual(thermo.Cp0.value_si, 33.3, 1)
        self.assertAlmostEqual(thermo.CpInf.value_si, 232.8, 1)
        self.assertEqual(len(thermo.Cpdata.value_si), 7)
