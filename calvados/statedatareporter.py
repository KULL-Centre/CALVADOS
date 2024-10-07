"""
statedatareporter.py: Outputs data about a simulation

This file is a modified version of OpenMM code from the OpenMM molecular simulation toolkit. It was originally modified by Zheng Gong in the mstk repository (https://github.com/z-gong/mstk).

The OpenMM molecular simulation toolkit originates from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2012-2021 Stanford University, the Authors, and Contributors.
Original Authors: Peter Eastman
Original Contributors: Robert McGibbon

Further modifications by Zheng Gong and Giulio Tesei.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from __future__ import absolute_import
from __future__ import print_function
__author__ = "Peter Eastman"
__version__ = "1.0"

try:
    import bz2
    have_bz2 = True
except: have_bz2 = False

try:
    import gzip
    have_gzip = True
except: have_gzip = False

import openmm as mm
import openmm.unit as unit
import math
import time
import numpy as np

class StateDataReporter(object):
    """StateDataReporter outputs information about a simulation, such as energy and temperature, to a file.

    This reporter is modified from the StateDataReporter shipped with OpenMM python API.
    It adds support for reporting the pressure tensor.

    To use it, create a StateDataReporter, then add it to the Simulation's list of reporters.  The set of
    data to write is configurable using boolean flags passed to the constructor.  By default the data is
    written in comma-separated-value (CSV) format, but you can specify a different separator to use.
    """

    def __init__(self, file, reportInterval, pressure_tensor=False, append=False, volume=0):
        """Create a StateDataReporter.

        Parameters
        ----------
        file : string or file
            The file to write to, specified as a file name or file object
        reportInterval : int
            The interval (in time steps) at which to write frames
        append : bool=False
            If true, append to an existing file.  This has two effects.  First,
            the file is opened in append mode.  Second, the header line is not
            written, since there is assumed to already be a header line at the
            start of the file.
        volume : float=0
            Periodic box volume
        """
        self._reportInterval = reportInterval
        self._openedFile = isinstance(file, str)
        if self._openedFile:
            if file.endswith('.npy'):
                if append:
                    self._pressure = np.load(file).tolist()
                else:
                    self._pressure = []
                self._out = file
            else:
                self._out = open(file, 'a' if append else 'w')
        else:
            self._out = file
        self._hasInitialized = False
        self._volume = volume
        self._needsPositions = True
        self._needsVelocities = False
        self._needsForces = True
        self._needEnergy = True
        self._includes = ['energy'] if self._needEnergy else []

    def describeNextReport(self, simulation):
        """Get information about the next report this object will generate.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for

        Returns
        -------
        tuple
            A five element tuple. The first element is the number of steps
            until the next report. The remaining elements specify whether
            that report will require positions, velocities, forces, and
            energies respectively.
        """
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, self._needsPositions, self._needsVelocities, self._needsForces, self._needEnergy)

    def report(self, simulation, state):
        """Generate a report.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        state : State
            The current state of the simulation
        """
        self._pressure.append(self._compute_pressure_tensor(simulation.context, state))

    def _compute_pressure_tensor(self, context, state):
        '''
        Compute the pressure tensor of a cuboidal system

        Parameters
        ----------
        context : mm.Context
        state : mm.State
        '''
        positions = state.getPositions(asNumpy=True)
        forces = state.getForces(asNumpy=True)

        p_kinetic = 2 * state.getKineticEnergy()

        p_virial = np.einsum('ij,ik->jk', positions, forces)[[0, 0, 0, 1, 1, 2],[0, 1, 2, 1, 2, 2]]

        return p_kinetic / unit.kilojoules_per_mole + p_virial

    def __del__(self):
        if self._openedFile:
            np.save(self._out,np.asarray(self._pressure)/3/self._volume*100/6.02214076)
