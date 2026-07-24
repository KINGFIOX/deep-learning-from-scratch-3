from typing import Optional, Union, cast

import numpy as np

ArrayOrScalar = Union[np.ndarray, np.generic, bool, int, float, complex]


class Variable:
    def __init__(self, data: Optional[np.ndarray]) -> None:
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func: "Function") -> None:
        self.creator = func

    def backward(self) -> None:
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)


def as_array(x: ArrayOrScalar) -> np.ndarray:
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, input: Variable) -> Variable:
        x = cast(np.ndarray, input.data)
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self)
        self.input = input
        self.output = output
        return output

    def forward(self, x: np.ndarray) -> ArrayOrScalar:
        raise NotImplementedError()

    def backward(self, gy: ArrayOrScalar) -> ArrayOrScalar:
        raise NotImplementedError()


class Square(Function):
    def forward(self, x: np.ndarray) -> ArrayOrScalar:
        y = x ** 2
        return y

    def backward(self, gy: ArrayOrScalar) -> ArrayOrScalar:
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x: np.ndarray) -> ArrayOrScalar:
        y = np.exp(x)
        return y

    def backward(self, gy: ArrayOrScalar) -> ArrayOrScalar:
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


def square(x: Variable) -> Variable:
    return Square()(x)


def exp(x: Variable) -> Variable:
    return Exp()(x)


x = Variable(np.array(0.5))
y = square(exp(square(x)))
y.backward()
print(x.grad)


x = Variable(np.array(1.0))  # OK
x = Variable(None)  # OK
x = Variable(1.0)  # NG
