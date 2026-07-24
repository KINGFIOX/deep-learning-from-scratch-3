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
        return output

    def forward(self, in_data: ArrayOrScalar) -> ArrayOrScalar:
        raise NotImplementedError()


class Square(Function):
    def forward(self, x: ArrayOrScalar) -> ArrayOrScalar:
        return x ** 2


x = Variable(np.array(10))
f = Square()
y = f(x)
print(type(y))
print(y.data)
