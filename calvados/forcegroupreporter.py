from openmm import unit

class ForceGroupReporter(object):
    def __init__(self, file, reportInterval, group, append=False):
        self._reportInterval = reportInterval
        self._group = group
        self._openedFile = isinstance(file, str)
        self._out = open(file, 'a' if append else 'w') if self._openedFile else file
        if not append:
            print('step\tenergy', file=self._out)
            self._out.flush()

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return {'steps': steps, 'periodic': None, 'include': []}

    def report(self, simulation, state):
        e = simulation.context.getState(getEnergy=True, groups=1 << self._group).getPotentialEnergy()
        print(f'{simulation.currentStep}\t{e.value_in_unit(unit.kilojoule_per_mole)}', file=self._out)
        self._out.flush()

    def __del__(self):
        if self._openedFile:
            self._out.close()
