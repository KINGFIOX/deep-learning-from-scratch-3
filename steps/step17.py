import weakref
from typing import Optional, Union, cast

import numpy as np

ArrayOrScalar = Union[np.ndarray, np.generic, bool, int, float, complex]
ArrayOrScalars = Union[ArrayOrScalar, tuple[ArrayOrScalar, ...]]


class Variable:
    def __init__(self, data: Optional[np.ndarray]) -> None:
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func: "Function") -> None:
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self) -> None:
        self.grad = None

    def backward(self) -> None:
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f: "Function") -> None:
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]  # output is weakref
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)


VariableOrVariables = Union[Variable, list[Variable]]


def as_array(x: ArrayOrScalar) -> np.ndarray:
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, *inputs: Variable) -> VariableOrVariables:
        xs = [cast(np.ndarray, x.data) for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        self.generation = max([x.generation for x in inputs])
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *xs: np.ndarray) -> ArrayOrScalars:
        raise NotImplementedError()

    def backward(self, *gys: ArrayOrScalar) -> ArrayOrScalars:
        raise NotImplementedError()


class Square(Function):
    def forward(self, x: np.ndarray) -> ArrayOrScalar:
        y = x ** 2
        return y

    def backward(self, gy: ArrayOrScalar) -> ArrayOrScalar:
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


def square(x: Variable) -> Variable:
    return cast(Variable, Square()(x))


for i in range(10):
    x = Variable(np.random.randn(10000))  # big data
    y = square(square(square(x)))
