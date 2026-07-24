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

        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *xs: np.ndarray) -> ArrayOrScalars:
        raise NotImplementedError()

    def backward(self, *gys: ArrayOrScalar) -> ArrayOrScalars:
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> ArrayOrScalar:
        y = x0 + x1
        return y


def add(x0: Variable, x1: Variable) -> Variable:
    return cast(Variable, Add()(x0, x1))


x0 = Variable(np.array(2))
x1 = Variable(np.array(3))
y = add(x0, x1)
print(y.data)
