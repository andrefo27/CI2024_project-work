import numpy as np

# ========= FLAG ========= #
USE_SAFE = False   # True = funzioni safe, False = funzioni numpy standard

# ========= SAFE FUNCTIONS ========= #
def safe_div(x, y):
    if np.isclose(y, 0):
        return np.nan
    return x / y

def safe_tan(x):
    if np.isclose(np.cos(x), 0):
        return np.nan
    return np.tan(x)

def safe_cot(x):
    if np.isclose(np.sin(x), 0):
        return np.nan
    return np.cos(x) / np.sin(x)

def safe_sqrt(x):
    if x < 0:
        return np.nan
    return np.sqrt(x)

def safe_ln(x):
    if x <= 0:
        return np.nan
    return np.log(x)

def safe_arccot(x):
    if np.isclose(x, 0):
        return np.pi / 2
    return np.arctan(1 / x)

def safe_exp(val):
    return np.exp(np.clip(val, -100, 350))

def safe_cosh(val):
    return np.cosh(np.clip(val, -200, 200))

def safe_sinh(val):
    return np.sinh(np.clip(val, -200, 200))

def cube(val):
    return np.power(np.clip(val, -1e20, 1e20), 3)

def sqr(val):
    return np.power(np.clip(val, -1e50, 1e50), 2)

# ========= SWITCHER ========= #
if USE_SAFE:
    my_div   = safe_div
    my_tan   = safe_tan
    my_cot   = safe_cot
    my_sqrt  = safe_sqrt
    my_ln    = safe_ln
    my_arccot= safe_arccot
    my_exp   = safe_exp
    my_cosh  = safe_cosh
    my_sinh  = safe_sinh
    my_cube  = cube
    my_sqr   = sqr
else:
    my_div   = lambda x, y: x / y
    my_tan   = np.tan
    my_cot   = lambda x: np.cos(x) / np.sin(x)
    my_sqrt  = np.sqrt
    my_ln    = np.log
    my_arccot= lambda x: np.arctan(1 / x)
    my_exp   = np.exp
    my_cosh  = np.cosh
    my_sinh  = np.sinh
    my_cube  = lambda v: np.power(v, 3)
    my_sqr   = lambda v: np.power(v, 2)


# Computed with 60% of the entire dataset
def f1(x: np.ndarray) -> np.ndarray:
    x0 = x[0, :]
    return np.sin(x0)

def f2_ev1(x: np.ndarray) -> np.ndarray:
    x0, x1, x2 = x[0, :], x[1, :], x[2, :]

    return ((x0 * (1166569.2224039633 -
                  ((407211.57237809076 * (x1 + x0)) *
                   ((x2 + x0) * 0.1751262493989698))))
            +
            (((407211.57237809076 * (2 * x1)) +
              (407211.57237809076 * (x2 + (2 * x0)))) +
             (407211.57237809076 * x2)))

def f2_ev2(x: np.ndarray) -> np.ndarray:
    x0, x1, x2 = x[0, :], x[1, :], x[2, :]

    return (
        np.arctan((2 * x0) + (0.98622902973299 + ((-0.9774299250715413 + x2) + x1)))
        *
        (3634475.3477245686 +
         (((x1 + x0) * 0.01830621799485299) *
          ((-0.9989511938091487 * (x0 + x2)) * 3537659.0406569946)))
    )

def f2_ev3(x: np.ndarray) -> np.ndarray:
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

def f3_ev1(x: np.ndarray) -> np.ndarray:
    x0, x1, x2 = x[0, :], x[1, :], x[2, :]

    return (
        ((my_sqr(x0) + 3.993555241630826) - (2.468855564282463 * x2))
        +
        ((my_sqr(x0) - (my_sqr(x1) * x1)) - x2)
    )

def f3_ev2(x: np.ndarray) -> np.ndarray:
    x0, x1, x2 = x[0, :], x[1, :], x[2, :]

    return (
        (2.002683911863346 * my_sqr(x0))
        +
        ((3.9850503589375528 - (x2 * 3.5009790345132275))
         +
         my_cube((-0.00023539136658632374 - x1)))
    )

def f4_ev1(x: np.ndarray) -> np.ndarray:
    x0, x1 = x[0, :], x[1, :]
    return (6.995299191648694 * np.cos(x1)) + (3.2803889063120284 - (0.09093106209030315 * x0))

def f4_ev2(x: np.ndarray) -> np.ndarray:
    x0, x1 = x[0, :], x[1, :]
    return (6.9600502069780825 * np.cos(x1)) + (3.2794524213733283 + (-0.09122965339457964 * x0))

def f5_ev1(x: np.ndarray) -> np.ndarray:
    x0, x1 = x[0, :], x[1, :]
    return (x1 + x0) * 0.00010064840791317857

def f5_ev2(x: np.ndarray) -> np.ndarray:
    x0, x1 = x[0, :], x[1, :]
    return my_exp(-10.593151254153158 + (np.sin(x0) * (x1 * -0.30136023861483663)))

def f6_ev1(x: np.ndarray) -> np.ndarray:
    x0, x1 = x[0, :], x[1, :]
    return (x1 * 1.6953937438370514) - (x0 * 0.6936705811025641)

def f6_ev2(x: np.ndarray) -> np.ndarray:
    x0, x1 = x[0, :], x[1, :]
    return (1.6947538279052852 * x1) - (0.6945673410208579 * x0)

def f7_ev1(x: np.ndarray) -> np.ndarray:
    x0, x1 = x[0, :], x[1, :]
    return my_exp(1.561295175294946 + (x1 * (x0 + -0.02613681253974187)))

def f7_ev2(x: np.ndarray) -> np.ndarray:
    x0, x1 = x[0, :], x[1, :]
    return my_exp(1.5497230604686836 + (x0 * x1))

def f8_ev1(x: np.ndarray) -> np.ndarray:
    x0, x1, x2, x3, x4, x5 = x[0, :], x[1, :], x[2, :], x[3, :], x[4, :], x[5, :]
    return (
        x0 +
        (4.675760748480646 * (x3 - ((np.cos(x2) + x5) + (my_cosh(x4) + (my_cosh(x4) * 4.571646229301208))))) +
        (my_cube(x5) +
         (((my_cube(x5) - x1) +
           (4.675760748480646 * ((my_sinh(x3) - (4.675760748480646 * (-0.8676042152839172 + x5))) - (2.1691631403452503 * my_cosh(x4))))) +
          ((my_cube(x5) + (my_sinh(x3) + (4.675760748480646 * my_cube(x5)))) +
           (4.675760748480646 * (my_sqr(x5) * my_cube(x5)))))))

def f8_ev2(x: np.ndarray) -> np.ndarray:
    x0, x1, x2, x3, x4, x5 = x[0, :], x[1, :], x[2, :], x[3, :], x[4, :], x[5, :]
    return (
        (((my_cube(x3) + (my_cube((x5 - 6.206303451447836)) + my_cube(x3))) +
          my_cube(((x4 + np.sin((-1.3420634973170156 * x5))) - 6.206303451447836))) +
         (my_cosh(x5) * ((my_cube((x5 - 0.002825691031909529)) -
                          (x4 + ((x5 - 6.206303451447836) + ((2 * x5) - x4)))) +
                         my_cube((x5 + np.sin((-1.3420634973170156 * x5))))))) -
        ((((my_cube(((x4 + 0.25332376515564414) + x4)) + -305.55726001262485) +
           my_exp(x4)) +
          ((((-0.8000949351095057 + (x4 * x4)) * x4) +
            my_cube(((x4 + 1.5485941972169874) - 6.206303451447836))) *
           ((0.36758101663471887 + (x4 + 0.943379279335353)) + 2))) +
         (((x1 + (my_cosh(x4) + (6.206303451447836 - my_cube(x3)))) +
           (x0 + (0.6927580940814835 + ((0.33963595881532704 + x4) +
                                        my_cube((-0.8174479516729702 - x5)))))) +
          (x5 + x2))))

def f8_ev3(x: np.ndarray) -> np.ndarray:
    x0, x1, x2, x3, x4, x5 = x[0, :], x[1, :], x[2, :], x[3, :], x[4, :], x[5, :]
    return (
        ((((((x4 * x4) * 0.4991427146821015) + np.cos(x4) + 0.21807710951788753) *
           -191.34488188898908) +
          (2 * (((my_cube(x3) + (x4 * x4)) - my_cosh(x4)) + (x4 * x4)))) -
         (-191.07040326540172 + np.cos(x2))) +
        ((((((-1.1397882197335478 + np.cos((-0.8652716641979434 * x4))) -
            -1.8176821373732027e-05) * -77.01579866243283) +
           (((0.9452511652132713 - my_sqr(x1)) + (my_cube(x3) + x1)) + (x4 * x4))) -
          my_cosh(x4)) +
         ((my_cube(x5) * my_sqr((2 * x5))) +
          (((11.636265720405797 + x0) - my_cosh(x4)) +
           (0.5358978385659909 + (my_sqr((x5 * x5)) * x5))))))

# Computed with all the dataset
def f2_all(x: np.ndarray) -> np.ndarray:
    x0, x1, x2 = x[0, :], x[1, :], x[2, :]
    return (
        (2 * np.arctan(((2 * x0) + (x2 + x1))))
        *
        (((x2 + x1) *
          (((x0 + x2) * -5509.0006723736315) *
           ((x0 * 0.5747860657142498) * x0)))
         + 1785299.4561500459)
    )

def f3_all(x: np.ndarray) -> np.ndarray:
    x0, x1, x2 = x[0, :], x[1, :], x[2, :]
    return (
        ((my_sqr(x0) - x2) -
         ((x2 * np.cosh(np.tanh(x1))) - 1.3302391133518534))
        +
        ((x1 * ((x0 * 3.723724201610524e-12) - my_sqr(x1)))
         -
         (((x2 - 0.6392235377956723) - my_sqr(x0)) + -2.039590593909345))
    )

def f4_all(x: np.ndarray) -> np.ndarray:
    x0, x1 = x[0, :], x[1, :]
    return (
        (np.cos(x1) + 0.4675685895554704)
        *
        (np.cos(np.tanh(x0)) + 6.3984015985878955)
    )

def f5_all(x: np.ndarray) -> np.ndarray:
    x0, x1 = x[0, :], x[1, :]
    return my_exp(
        (-10.593151254153158 + (np.sin(x0) * (x1 * -0.30136023861483663)))
    )

def f6_all(x: np.ndarray) -> np.ndarray:
    x0, x1 = x[0, :], x[1, :]
    return ((1.6942484016981965 * x1) - (0.694681289433442 * x0))

def f7_all(x: np.ndarray) -> np.ndarray:
    x0, x1 = x[0, :], x[1, :]
    return (
        (my_exp(((x0 * x1) * 1.0797632614418142)))
        * 3.665180185745142
    )

def f8_all(x: np.ndarray) -> np.ndarray:
    x0, x1, x2, x3, x4, x5 = x[0, :], x[1, :], x[2, :], x[3, :], x[4, :], x[5, :]
    return (
        (my_sqr((2.0 * x5)) * (my_cube(x5) + 0.004274202671157051))
        +
        ((my_sqr(x5) * my_cube(x5))
         +
         ((my_sqr((x4 * (2.0 * x4))) * -0.9626766210100366)
          +
          (((((my_cosh(x1) * -0.828060103226649) + (x2 - -0.7769462579782747))
             + (my_cube(x3) + ((0.02837821037606747 * x0) + x0)))
            - (x2 - -1.8241925696267818))
           +
           ((my_sqr((2.0 * x4)) * -0.7506875922133385)
            + (my_cube(x3) + (my_cube(x3) + 2.028632985742853))))))
    )


# EVALUATION
if __name__ == "__main__":

    func_values = {
        "f1": 1,
        "f2_ev1": 2,
        "f2_ev2": 2,
        "f2_ev3": 2,
        "f3_ev1": 3,
        "f3_ev2": 3,
        "f4_ev1": 4,
        "f4_ev2": 4,
        "f5_ev1": 5,
        "f5_ev2": 5,
        "f6_ev1": 6,
        "f6_ev2": 6,
        "f7_ev1": 7,
        "f7_ev2": 7,
        "f8_ev1": 8,
        "f8_ev2": 8,
        "f8_ev3": 8,
        "f2_all": 2,
        "f3_all": 3,
        "f4_all": 4,
        "f5_all": 5,
        "f6_all": 6,
        "f7_all": 7,
        "f8_all": 8
    }

    for func_name, problem_id in func_values.items():
        data = np.load(f"../data/problem_{problem_id}.npz")

        x = data['x']
        y = data['y']

        TRAIN_SIZE = 1

        # decide how many data to train the model (0.6 --> 60% of the data, 1 --> 100% of the data)
        train_size = int(x.shape[1] * TRAIN_SIZE)

        # (n_features, n_samples)
        x = x[:, :train_size]

        y = y[:train_size]

        func = globals()[func_name]

        y_pred = func(x)

        try:
            mse = np.mean((y_pred - y) ** 2)
            print(f"MSE for problem {problem_id} using {func_name}: {mse:7f}")
        except Exception as e:
            print(f"Problem {problem_id} not solved with {func_name}.")
            print(f"Exception: {e}")


""" 
# use this code for the evaluation of s324581.py and s324581_datasplit.py
def symreg(problem_id):
    data = np.load(f"data/problem_{problem_id}.npz")

    x = data['x']
    y = data['y']

    func = [0, f1, f2, f3, f4, f5, f6, f7, f8][problem_id]

    y_pred = func(x)

    try:
        mse = np.mean((y_pred - y) ** 2)
        print(f"MSE for problem {problem_id} using {func}: {mse:7f}")
    except Exception as e:
        print(f"Problem {problem_id} not solved with {func}.")
        print(f"Exception: {e}")

for i in range(1,9):
    symreg(i) """