import numpy as np
# no clip and safe functions as in training evaluation

def f1(x: np.ndarray) -> np.ndarray:
    x0 = x[0, :]
    return np.sin(x0)

def f2(x: np.ndarray) -> np.ndarray:
    x0, x1, x2 = x[0, :], x[1, :], x[2, :]
    return (
        (2 * np.arctan(((2 * x0) + (x2 + x1))))
        *
        (((x2 + x1) *
          (((x0 + x2) * -5509.0006723736315) *
           ((x0 * 0.5747860657142498) * x0)))
         + 1785299.4561500459)
    )

def f3(x: np.ndarray) -> np.ndarray:
    x0, x1, x2 = x[0, :], x[1, :], x[2, :]

    my_sqr   = lambda v: np.power(v, 2)
    return (
        ((my_sqr(x0) - x2) -
         ((x2 * np.cosh(np.tanh(x1))) - 1.3302391133518534))
        +
        ((x1 * ((x0 * 3.723724201610524e-12) - my_sqr(x1)))
         -
         (((x2 - 0.6392235377956723) - my_sqr(x0)) + -2.039590593909345))
    )

def f4(x: np.ndarray) -> np.ndarray:
    x0, x1 = x[0, :], x[1, :]
    return (
        (np.cos(x1) + 0.4675685895554704)
        *
        (np.cos(np.tanh(x0)) + 6.3984015985878955)
    )

def f5(x: np.ndarray) -> np.ndarray:
    x0, x1 = x[0, :], x[1, :]
    return np.exp(
        (-10.593151254153158 + (np.sin(x0) * (x1 * -0.30136023861483663)))
    )

def f6(x: np.ndarray) -> np.ndarray:
    x0, x1 = x[0, :], x[1, :]
    return ((1.6942484016981965 * x1) - (0.694681289433442 * x0))

def f7(x: np.ndarray) -> np.ndarray:
    x0, x1 = x[0, :], x[1, :]
    return (
        (np.exp(((x0 * x1) * 1.0797632614418142)))
        * 3.665180185745142
    )

def f8(x: np.ndarray) -> np.ndarray:
    x0, x1, x2, x3, x4, x5 = x[0, :], x[1, :], x[2, :], x[3, :], x[4, :], x[5, :]
    my_cube  = lambda v: np.power(v, 3)
    my_sqr   = lambda v: np.power(v, 2)

    return (
        (my_sqr((2.0 * x5)) * (my_cube(x5) + 0.004274202671157051))
        +
        ((my_sqr(x5) * my_cube(x5))
         +
         ((my_sqr((x4 * (2.0 * x4))) * -0.9626766210100366)
          +
          (((((np.cosh(x1) * -0.828060103226649) + (x2 - -0.7769462579782747))
             + (my_cube(x3) + ((0.02837821037606747 * x0) + x0)))
            - (x2 - -1.8241925696267818))
           +
           ((my_sqr((2.0 * x4)) * -0.7506875922133385)
            + (my_cube(x3) + (my_cube(x3) + 2.028632985742853))))))
    )