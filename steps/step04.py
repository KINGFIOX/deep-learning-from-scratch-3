from collections.abc import Callable
from typing import Union

import numpy as np

ArrayOrScalar = Union[np.ndarray, np.generic, bool, int, float, complex]


class Variable:
    def __init__(self, data: ArrayOrScalar) -> None:
        self.data = data


class Function:
    def __call__(self, input: Variable) -> Variable:
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        self.input = input
        self.output = output
        return output

    def forward(self, x: ArrayOrScalar) -> ArrayOrScalar:
        raise NotImplementedError()


class Square(Function):
    def forward(self, x: ArrayOrScalar) -> ArrayOrScalar:
        return x ** 2


class Exp(Function):
    def forward(self, x: ArrayOrScalar) -> ArrayOrScalar:
        return np.exp(x)


def numerical_diff(
    f: Callable[[Variable], Variable], x: Variable, eps: float = 1e-4
) -> ArrayOrScalar:
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


f = Square()
x = Variable(np.array(2.0))
dy = numerical_diff(f, x)
print(dy)


def f(x: Variable) -> Variable:
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))


x = Variable(np.array(0.5))
dy = numerical_diff(f, x)
print(dy)
