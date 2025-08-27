import numpy as np
# no clip and safe functions as in training evaluation

def f1(x: np.ndarray) -> np.ndarray:
    x0 = x[0, :]
    return np.sin(x0)

def f2(x: np.ndarray) -> np.ndarray:
    x0, x1, x2 = x[0, :], x[1, :], x[2, :]

    return (
        (x0 +
         ((-0.019111713360799722 *
           ((x0 + 0.9840486856017361) * 3519078.1007138137)) *
          ((x2 + x1) * (x0 + -1.3856915093946087))))
        +
        (np.arctan(((1.3787422579085915 * x0) + (x1 + x2))) *
         3519078.1007138137)
    )

def f3(x: np.ndarray) -> np.ndarray:
    x0, x1, x2 = x[0, :], x[1, :], x[2, :]

    my_sqr = lambda v: np.power(v, 2)
    my_cube = lambda v: np.power(v, 3)

    return (
        (2.002683911863346 * my_sqr(x0))
        +
        ((3.9850503589375528 - (x2 * 3.5009790345132275))
         +
         my_cube((-0.00023539136658632374 - x1)))
    )

def f4(x: np.ndarray) -> np.ndarray:
    x0, x1 = x[0, :], x[1, :]
    return (6.995299191648694 * np.cos(x1)) + (3.2803889063120284 - (0.09093106209030315 * x0))

def f5(x: np.ndarray) -> np.ndarray:
    x0, x1 = x[0, :], x[1, :]
    return np.exp(-10.593151254153158 + (np.sin(x0) * (x1 * -0.30136023861483663)))

def f6(x: np.ndarray) -> np.ndarray:
    x0, x1 = x[0, :], x[1, :]
    return (1.6947538279052852 * x1) - (0.6945673410208579 * x0)

def f7(x: np.ndarray) -> np.ndarray:
    x0, x1 = x[0, :], x[1, :]
    return np.exp(1.5497230604686836 + (x0 * x1))

def f8(x: np.ndarray) -> np.ndarray:
    x0, x1, x2, x3, x4, x5 = x[0, :], x[1, :], x[2, :], x[3, :], x[4, :], x[5, :]

    my_sqr = lambda v: np.power(v, 2)
    my_cube = lambda v: np.power(v, 3)

    return (
        ((((((x4 * x4) * 0.4991427146821015) + np.cos(x4) + 0.21807710951788753) *
           -191.34488188898908) +
          (2 * (((my_cube(x3) + (x4 * x4)) - np.cosh(x4)) + (x4 * x4)))) -
         (-191.07040326540172 + np.cos(x2))) +
        ((((((-1.1397882197335478 + np.cos((-0.8652716641979434 * x4))) -
            -1.8176821373732027e-05) * -77.01579866243283) +
           (((0.9452511652132713 - my_sqr(x1)) + (my_cube(x3) + x1)) + (x4 * x4))) -
          np.cosh(x4)) +
         ((my_cube(x5) * my_sqr((2 * x5))) +
          (((11.636265720405797 + x0) - np.cosh(x4)) +
           (0.5358978385659909 + (my_sqr((x5 * x5)) * x5))))))