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
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                x.grad = gx

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


class Square(Function):
    def forward(self, x: np.ndarray) -> ArrayOrScalar:
        y = x ** 2
        return y

    def backward(self, gy: ArrayOrScalar) -> ArrayOrScalar:
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


def square(x: Variable) -> Variable:
    f = Square()
    return cast(Variable, f(x))


class Add(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> ArrayOrScalar:
        y = x0 + x1
        return y

    def backward(
        self, gy: ArrayOrScalar
    ) -> tuple[ArrayOrScalar, ArrayOrScalar]:
        return gy, gy


def add(x0: Variable, x1: Variable) -> Variable:
    return cast(Variable, Add()(x0, x1))


# additional example from the original book
class Mul(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> ArrayOrScalar:
        y = x0 * x1
        return y

    def backward(
        self, gy: ArrayOrScalar
    ) -> tuple[ArrayOrScalar, ArrayOrScalar]:
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0


def mul(x0: Variable, x1: Variable) -> Variable:
    return cast(Variable, Mul()(x0, x1))


x = Variable(np.array(2.0))
y = Variable(np.array(3.0))

z = add(square(x), square(y))
z.backward()
print(z.data)
print(x.grad)
print(y.grad)


x = Variable(np.array(2.0))
y = Variable(np.array(3.0))

z = mul(x, y)
z.backward()
print(z.data)
print(x.grad)
print(y.grad)
