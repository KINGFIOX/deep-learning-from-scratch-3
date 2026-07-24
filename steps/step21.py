import contextlib
import weakref
from collections.abc import Iterator
from typing import Optional, Union, cast, overload

import numpy as np

Scalar = Union[np.generic, bool, int, float, complex]
ArrayOrScalar = Union[np.ndarray, Scalar]
ArrayOrScalars = Union[ArrayOrScalar, tuple[ArrayOrScalar, ...]]


class Config:
    enable_backprop = True


@contextlib.contextmanager
def using_config(name: str, value: bool) -> Iterator[None]:
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad() -> contextlib.AbstractContextManager[None]:
    return using_config('enable_backprop', False)


class Variable:
    __array_priority__ = 200

    def __init__(
        self, data: Optional[np.ndarray], name: Optional[str] = None
    ) -> None:
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

    def set_creator(self, func: "Function") -> None:
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self) -> None:
        self.grad = None

    def backward(self, retain_grad: bool = False) -> None:
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

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None  # y is weakref


VariableOrArray = Union[Variable, np.ndarray]
Operand = Union[Variable, ArrayOrScalar]
VariableOrVariables = Union[Variable, list[Variable]]


def as_variable(obj: VariableOrArray) -> Variable:
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


@overload
def as_array(x: ArrayOrScalar) -> np.ndarray:
    ...


@overload
def as_array(x: Variable) -> Variable:
    ...


def as_array(x: Operand) -> VariableOrArray:
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, *inputs: VariableOrArray) -> VariableOrVariables:
        inputs = [as_variable(x) for x in inputs]

        xs = [cast(np.ndarray, x.data) for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
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


class Add(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> ArrayOrScalar:
        y = x0 + x1
        return y

    def backward(
        self, gy: ArrayOrScalar
    ) -> tuple[ArrayOrScalar, ArrayOrScalar]:
        return gy, gy


def add(x0: VariableOrArray, x1: Operand) -> Variable:
    x1 = as_array(x1)
    return cast(Variable, Add()(x0, x1))


class Mul(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> ArrayOrScalar:
        y = x0 * x1
        return y

    def backward(
        self, gy: ArrayOrScalar
    ) -> tuple[ArrayOrScalar, ArrayOrScalar]:
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0


def mul(x0: VariableOrArray, x1: Operand) -> Variable:
    x1 = as_array(x1)
    return cast(Variable, Mul()(x0, x1))


Variable.__add__ = add
Variable.__radd__ = add
Variable.__mul__ = mul
Variable.__rmul__ = mul

x = Variable(np.array(2.0))
y = x + np.array(3.0)
print(y)

y = x + 3.0
print(y)

y = 3.0 * x + 1.0
print(y)
