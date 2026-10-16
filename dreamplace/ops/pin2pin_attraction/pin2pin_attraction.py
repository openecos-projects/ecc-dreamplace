##
# @file   pin2pin_attraction.py (modified weighted_average_wirelength.py)
# @author Yibo Lin
# @date   Jun 2018
# @brief  Compute weighted-average wirelength according to e-place
#

import time
import torch
from torch import nn
from torch.autograd import Function
import logging
import numpy as np

import dreamplace.ops.pin2pin_attraction.pin2pin_attraction_cpp as pin2pin_attraction_cpp
import dreamplace.configure as configure
if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    import dreamplace.ops.pin2pin_attraction.pin2pin_attraction_cuda as pin2pin_attraction_cuda
import pdb

logger = logging.getLogger(__name__)


class Pin2PinAttractionFunction(Function):
    """
    @brief compute Pin2PinAttraction.
    """
    @staticmethod
    def forward(
        ctx,
        pin_pos,
        pin_mask,
        pin2pin_net_weight,
        pairs, weights, length
    ):
        """
        @param pin_pos pin location (x array, y array)
               length of #nets+1, the last entry is #pins
        @param net_weights weight of nets
        """
        tt = time.time()
        ctx.pin_pos = pin_pos
        ctx.pin_mask = pin_mask
        ctx.pin2pin_net_weight = pin2pin_net_weight
        ctx.pairs = pairs
        ctx.weights = weights
        ctx.length = length
        if pin_pos.is_cuda:
            if length[0] == 0:
                return torch.zeros(1).to(pin_pos.device)
            func = pin2pin_attraction_cuda.forward
            output = func(
                pin_pos.view(pin_pos.numel()),
                pairs[:2*length[0]],
                weights[:length[0]]
            )
        else:
            func = pin2pin_attraction_cpp.forward
            output = func(
                pin_pos.view(pin_pos.numel()),
                pin2pin_net_weight
            )

        
        if pin_pos.is_cuda:
            torch.cuda.synchronize()
        logger.debug("Pin2PinAttraction forward %.3f ms" %
                     ((time.time() - tt) * 1000))
        return output[0]

    @staticmethod
    def backward(ctx, grad_pos):
        tt = time.time()
        if grad_pos.is_cuda:
            if ctx.length[0] == 0:
                return None, None, None, None, None, None
            func = pin2pin_attraction_cuda.backward
            output = func(
                grad_pos,
                ctx.pin_pos,
                ctx.pairs[:2*ctx.length[0]],
                ctx.weights[:ctx.length[0]]
            )
        else:
            func = pin2pin_attraction_cpp.backward
            output = func(
                grad_pos,
                ctx.pin_pos,
                ctx.pin2pin_net_weight
            )
        if grad_pos.is_cuda:
            torch.cuda.synchronize()
        logger.debug("Pin2PinAttraction backward %.3f ms" %
                     ((time.time() - tt) * 1000))
        output[:output.numel() // 2].masked_fill_(ctx.pin_mask, 0.0)
        output[output.numel() // 2:].masked_fill_(ctx.pin_mask, 0.0)
        # print("pin2pin grad: ", torch.sum(output.norm()), output, torch.norm(output, p=float('inf')))

        return output, None, None, None, None, None



class Pin2PinAttraction(nn.Module):
    """
    @brief Compute Pin2PinAttraction.
    CPU only supports net-by-net algorithm.
    GPU supports three algorithms: net-by-net, atomic, merged.
    Different parameters are required for different algorithms.
    """
    def __init__(self,
                 pin_mask,
                 pin2pin_net_weight,
                 pairs,
                 weights,
                 length
    ):
        """
        @brief initialization
        @param pin_direction the smaller, the closer to HPWL
        """
        super(Pin2PinAttraction, self).__init__()
        self.pin_mask = pin_mask
        self.pin2pin_net_weight = pin2pin_net_weight
        self.pairs = pairs
        self.weights = weights
        self.length = length

    def forward(self, pin_pos):
        return Pin2PinAttractionFunction.apply(
            pin_pos,
            self.pin_mask,
            self.pin2pin_net_weight,
            self.pairs,
            self.weights,
            self.length
        )
        

