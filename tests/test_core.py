import inspect
import unittest

import numpy as np

from dezero import Variable


class TestArrayGrad(unittest.TestCase):

    def test_grad_is_ndarray(self):
        x = Variable(np.array(2.0))
        y = x ** 2

        y.backward()

        self.assertIsInstance(x.grad, np.ndarray)
        self.assertEqual(x.grad, np.array(4.0))

    def test_grad_accumulation_uses_arrays(self):
        x = Variable(np.array(3.0))
        y = x * x + x

        y.backward()

        self.assertIsInstance(x.grad, np.ndarray)
        self.assertEqual(x.grad, np.array(7.0))

    def test_retain_grad_keeps_intermediate_array(self):
        x = Variable(np.array(2.0))
        y = x ** 2
        z = y ** 2

        z.backward(retain_grad=True)

        self.assertIsInstance(y.grad, np.ndarray)
        self.assertEqual(y.grad, np.array(8.0))

    def test_backward_signature_only_exposes_retain_grad(self):
        parameters = list(inspect.signature(Variable.backward).parameters)
        self.assertEqual(parameters, ['self', 'retain_grad'])


if __name__ == '__main__':
    unittest.main()
