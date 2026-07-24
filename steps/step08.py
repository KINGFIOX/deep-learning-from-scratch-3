from typing import Union

import numpy as np

ArrayOrScalar = Union[np.ndarray, np.generic, bool, int, float, complex]


class Variable:
    def __init__(self, data: ArrayOrScalar) -> None:
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func: "Function") -> None:
        self.creator = func

    def backward(self) -> None:
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()  # 1. Get a function
            x, y = f.input, f.output  # 2. Get the function's input/output
            x.grad = f.backward(y.grad)  # 3. Call the function's backward

            if x.creator is not None:
                funcs.append(x.creator)


class Function:
    def __call__(self, input: Variable) -> Variable:
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)
        self.input = input
        self.output = output
        return output

    def forward(self, x: ArrayOrScalar) -> ArrayOrScalar:
        raise NotImplementedError()

    def backward(self, gy: ArrayOrScalar) -> ArrayOrScalar:
        raise NotImplementedError()


class Square(Function):
    def forward(self, x: ArrayOrScalar) -> ArrayOrScalar:
        y = x ** 2
        return y

    def backward(self, gy: ArrayOrScalar) -> ArrayOrScalar:
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x: ArrayOrScalar) -> ArrayOrScalar:
        y = np.exp(x)
        return y

    def backward(self, gy: ArrayOrScalar) -> ArrayOrScalar:
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

# backward
y.grad = np.array(1.0)
y.backward()
print(x.grad)
