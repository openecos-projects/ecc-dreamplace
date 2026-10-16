"""CUDA Pin2Pin correctness and bitwise repeatability regressions."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dreamplace.ops.pin2pin_attraction import pin2pin_attraction_cuda as native
from dreamplace.ops.pin2pin_attraction.pin2pin_attraction import Pin2PinAttractionFunction


def inputs(dtype, pins=4096, count=32834):
    generator = torch.Generator().manual_seed(3000)
    pos = torch.rand(2 * pins, generator=generator, dtype=dtype) * 7000
    pairs = torch.randint(pins, (2 * count,), generator=generator, dtype=torch.int32)
    weights = 10 + 40 * torch.rand(count, generator=generator, dtype=dtype)
    return pos.cuda(), pairs.cuda(), weights.cuda()


def reference(pos, pairs, weights, upstream):
    # CPU float64 autograd is independent of the CUDA reduction implementation.
    xy = pos.cpu().double().reshape(2, -1).requires_grad_()
    ends = pairs.cpu().long().reshape(-1, 2)
    delta = xy[:, ends[:, 0]] - xy[:, ends[:, 1]]
    loss = (weights.cpu().double() * delta.square().sum(0)).sum()
    gradient, = torch.autograd.grad(loss * upstream, xy)
    return loss.detach(), gradient.flatten()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class Pin2PinAttractionTest(unittest.TestCase):
    def test_repeatability(self):
        old = torch.are_deterministic_algorithms_enabled()
        self.addCleanup(torch.use_deterministic_algorithms, old)
        torch.use_deterministic_algorithms(True)
        for dtype in (torch.float32, torch.float64):
            pos, pairs, weights = inputs(dtype)
            upstream = pos.new_tensor(1.0)
            loss = native.forward(pos, pairs, weights)[0].cpu()
            gradient = native.backward(upstream, pos, pairs, weights).cpu()
            for _ in range(100):
                self.assertTrue(torch.equal(loss, native.forward(pos, pairs, weights)[0].cpu()))
                self.assertTrue(torch.equal(gradient, native.backward(upstream, pos, pairs, weights).cpu()))

    def test_reference_and_collisions(self):
        for dtype in (torch.float32, torch.float64):
            for count in (0, 1, 255, 256, 257, 32834):
                with self.subTest(dtype=dtype, count=count):
                    pos, pairs, weights = inputs(dtype, pins=127, count=count)
                    # High fan-in, reversed duplicates, self-pairs and cancellation.
                    pairs[::2] = 0
                    if count >= 4:
                        pairs[:8] = pairs.new_tensor([0, 1, 1, 0, 0, 0, 0, 1])
                        weights[:4] = weights.new_tensor([30, 30, 10, -30])
                    expected_loss, expected_grad = reference(pos, pairs, weights, -0.375)
                    loss = native.forward(pos, pairs, weights)[0].cpu().double()
                    gradient = native.backward(pos.new_tensor(-0.375), pos, pairs, weights).cpu().double()
                    rtol, atol = (2e-5, 0.1) if dtype == torch.float32 else (1e-12, 1e-7)
                    torch.testing.assert_close(loss, expected_loss, rtol=rtol, atol=atol)
                    torch.testing.assert_close(gradient, expected_grad, rtol=rtol, atol=atol)

    def test_gradcheck_and_mask(self):
        pos, pairs, weights = inputs(torch.float64, pins=8, count=19)
        pos = (pos / 7000).requires_grad_()
        mask = torch.zeros(8, dtype=torch.bool, device=pos.device)
        length = torch.tensor([weights.numel()], dtype=torch.int32)
        def op(value):
            return Pin2PinAttractionFunction.apply(value, mask, {}, pairs, weights, length)
        self.assertTrue(torch.autograd.gradcheck(op, (pos,), eps=1e-6, atol=1e-5, rtol=1e-5))
        expected = native.backward(pos.new_tensor(1), pos, pairs, weights)
        mask[::2] = True
        actual, = torch.autograd.grad(op(pos), pos)
        expected.reshape(2, -1)[:, mask] = 0
        self.assertTrue(torch.equal(actual, expected))

    def test_current_stream(self):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            pos, pairs, weights = inputs(torch.float64, pins=257, count=1025)
            torch.cuda._sleep(10_000_000)
            pos = pos * 0.25
            loss = native.forward(pos, pairs, weights)[0]
            gradient = native.backward(pos.new_tensor(2), pos, pairs, weights)
            # Consumers must also be correctly ordered on the caller's stream.
            loss, gradient = loss.clone(), gradient.clone()
        stream.synchronize()
        expected_loss, expected_grad = reference(pos, pairs, weights, 2)
        torch.testing.assert_close(loss.cpu(), expected_loss, rtol=1e-12, atol=1e-7)
        torch.testing.assert_close(gradient.cpu(), expected_grad, rtol=1e-12, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
