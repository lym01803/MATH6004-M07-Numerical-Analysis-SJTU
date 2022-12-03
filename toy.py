import math 
from typing import Callable, Tuple

import numpy as np

def bi_search(f: Callable[[float], float], x0: float, x1: float, eps: float = 1e-4):
    fx0 = f(x0)
    fx1 = f(x1)
    while math.fabs(x1 - x0) > eps:
        m = 0.5 * (x1 + x0)
        fm = f(m)
        print(f'x0 = {x0}, m = {m}, x1 = {x1}, f(x0) = {fx0}, f(m) = {fm}, f(x1) = {fx1}')
        if fm * fx0 < 0:
            x1 = m
            fx1 = fm 
        else:
            x0 = m
            fx0 = fm 
    print(f'final: x0 = {x0}, x1 = {x1}')
    return x0
        

def iter_method(f: Callable[[float], float], x0: float, eps: float = 1e-6, max_iter = 10):
    x1 = f(x0)
    i = 0
    while math.fabs(x1 - x0) > eps or i < max_iter:
        print(f'x = {x0}, f(x) = {x1}')
        x0 = x1 
        x1 = f(x0)
        i += 1
    print(f'x = {x0}, f(x) = {x1}')
    return x1


def Newton(f: Callable[[Tuple], Tuple], x0: Tuple, eps: float = 5e-5):
    x1 = f(x0)
    while math.fabs(x1[0] - x0[0]) > eps:
        x0 = x1 
        x1 = f(x0)
    return x1


def f1(x: Tuple):
    x0 = x[0]
    y = x0 - (x0 ** 3 - 3 * x0 - 1) / (3 * x0 ** 2 - 3)
    print(f'x0 = {x0}, x1 = phi(x0) = {y}')
    return (y,)


def f2(f: Callable[[float], float], x: Tuple):
    x0, f0, x1, f1 = x
    f01 = (f1 - f0) / (x1 - x0)
    x2 = x1 - f1 / f01
    f2 = f(x2)
    print(f'x0 = {x0}, x1 = {x1}, f(x0) = {f0}, f(x1) = {f1}, f[x0, x1] = {f01}, x2 = {x2}')
    return (x1, f1, x2, f2)

# bi_search(lambda x: x ** 2 - x - 1, 0., 2., 0.05)
# iter_method(lambda x: 1 + 1 / (x ** 2), 1.5, max_iter=30)
# iter_method(lambda x : math.atan(x) + math.pi, 4.5, 1e-6)


def f3(f: Callable[[float], float], x: Tuple):
    x0, f0, x1, f1, x2, f2 = x 
    f21 = (f1 - f2) / (x1 - x2)
    f20 = (f0 - f2) / (x0 - x2)
    f210 = (f20 - f21) / (x0 - x1)
    w = f21 + f210 * (x2 - x1)
    delta = math.sqrt(w * w - 4 * f2 * f210)
    x31 = x2 - 2 * f2 / (w + delta)
    x32 = x2 - 2 * f2 / (w - delta)
    if math.fabs(x31 - x2) < math.fabs(x32 - x2):
        x3 = x31
    else:
        x3 = x32 
    print(f'x0 = {x0}, x1 = {x1}, x2 = {x2}, f210 = {f210} x31 = {x31}, x32 = {x32}, x3 = {x3}')
    return (x1, f1, x2, f2, x3, f(x3))


print(Newton(f1, (2.,)))
print("================")
f = lambda x: x ** 3 - 3 * x - 1
print(Newton(lambda x: f2(f, x), (2., f(2.), 1.9, f(1.9))))
print("================")
print(Newton(lambda x: f3(f, x), (1., f(1.), 3., f(3.), 2., f(2.))))
