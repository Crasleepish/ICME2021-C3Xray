import sys

class ProgressBar:
    def __init__(self, steps):
        self.steps = steps
        self.toolbar_width = 100

    def printBar(self):
        c = self.steps % 1000 * self.toolbar_width // 1000
        sys.stdout.write('\r')
        sys.stdout.write('[%s%s]' % ('>' * c, '.' * (self.toolbar_width - c)))
        sys.stdout.write(str(self.steps))
        sys.stdout.flush()

    def step_forward(self):
        self.steps += 1
        self.printBar()