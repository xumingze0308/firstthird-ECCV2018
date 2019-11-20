import torch
import numpy as np

# Ref: https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
# Ref: https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/models/fcn32s.py
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels,
                       kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()

def compute_iou(inputs, targets):
    result = 0.0
    if type(inputs) == type(targets) == torch.autograd.Variable:
        inputs = inputs.cpu().data.numpy()
        targets = targets.cpu().data.numpy()

    if type(inputs) == type(targets) == np.ndarray:
        for input, target in zip(inputs, targets):
            interest = float(np.sum((target != 0) & (target == input)))
            union = float(np.sum((target != 0) | (input !=0)))
            assert union >= 1
            result += interest / union
        return result / inputs.shape[0]
    else:
        raise(RuntimeError('Usage: IoU(inputs, targets), and inputs and targets '
                           'should be either torch.autograd.Variable or numpy.ndarray.'))
