from ignite.metrics import Accuracy, Metric
import torch
# Custom metric class
class CustomMetric(Metric):
    def __init__(self, output_transform=lambda x: x, device=None):
        self._sum = None
        self._num_examples = None
        super(CustomMetric, self).__init__(output_transform=output_transform, device=device)

    def reset(self):
        self._sum = 0.0
        self._num_examples = 0

    def update(self, output):
        y_pred, y = output[0], output[1]
        self._sum += torch.sum(y_pred)
        self._num_examples += y_pred.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('CustomMetric must have at least one example before it can be computed')
        return self._sum / self._num_examples