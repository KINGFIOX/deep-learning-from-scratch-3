from collections.abc import Sequence
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
    def __call__(self, inputs: Sequence[Variable]) -> list[Variable]:
        xs = [cast(np.ndarray, x.data) for x in inputs]
        ys = self.forward(xs)
        outputs = [Variable(as_array(y)) for y in ys]  # Wrap data

        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs

    def forward(self, xs: Sequence[np.ndarray]) -> tuple[ArrayOrScalar, ...]:
        raise NotImplementedError()

    def backward(
        self, gys: Sequence[ArrayOrScalar]
    ) -> tuple[ArrayOrScalar, ...]:
        raise NotImplementedError()


class Add(Function):
    def forward(self, xs: Sequence[np.ndarray]) -> tuple[ArrayOrScalar, ...]:
        x0, x1 = xs
        y = x0 + x1
        return (y,)


xs = [Variable(np.array(2)), Variable(np.array(3))]
f = Add()
ys = f(xs)
y = ys[0]
print(y.data)
