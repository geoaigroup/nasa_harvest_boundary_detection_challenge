import numpy as np
import torch

# meters
class Meter(object):
    def reset(self):
        pass

    def update(self, value):
        pass

    def get_update(self):
        pass

    def set_name(self, name):
        self.name = f'{name}_{self.kind}'


class AverageMeter(Meter):
    def __init__(self):
        super(AverageMeter, self).__init__()
        self.reset()
        self.kind = 'avg'

    def reset(self):
        self.value = 0.0
        self.average = 0.0
        self.count = 0.0

    def update(self, value):
        self.count += 1
        self.value = value
        self.average = ((self.average * (self.count - 1)) + self.value) / float(self.count)

    def update_all(self, values):
        l = len(values)
        self.sum = np.sum(values)
        self.count += l
        self.average = ((self.average * (self.count - l)) + self.sum) / float(self.count)

    def get_update(self):
        return self.average


def meter_dict(channel_names):
    d = {channel_name : AverageMeter() for channel_name in channel_names}
    return d