import matplotlib.pyplot as plt
import numpy as np
import math


def alpha_n(v):
    return (0.1 - 0.01 * v) / (math.e ** (1. - 0.1 * v) - 1.)


def beta_n(v):
    return 0.125 * math.e ** (-1. * v / 80.)


def alpha_m(v):
    return (2.5 - 0.1 * v) / (math.e ** (2.5 - 0.1 * v) - 1.)


def beta_m(v):
    4. * math.e ** (-1. * v / 18.)


def alpha_h(v):
    0.07 * math.e ** (-1. * v / 20.)


def beta_h(v):
    1. / (math.e ** (3. - 0.1 * v) + 1.)


def channel_opening_prob_diff_equation(alpha, beta, cur_voltage, cur_value):
    return alpha(cur_voltage) * (1. - cur_value) - beta(cur_voltage) * cur_value


def channel_opening_prob(alpha, beta, cur_voltage, cur_prob, delta_t):
    diff_equation = lambda y, x: channel_opening_prob_diff_equation(alpha, beta, cur_voltage, y)
    return runge_kutta_method(diff_equation, cur_prob, 0, delta_t)


def channel_opening_prob_at_rest(alpha, beta, voltage_at_rest):
    return alpha(voltage_at_rest) / (alpha(voltage_at_rest) + beta(voltage_at_rest))


def current(t):
    return 0.


def voltage_diff_equation(cur_voltage, I, m, n, h):
    C = 1.

    Gn = 120.
    En = 115.

    Gk = 36.
    Ek = -12.

    Gl = 0.3
    El = 10.6

    V = cur_voltage

    differential_value = 1. / C * (I - (Gn * m ** 3 * h * (V - En) + Gk * n ** 4 * (V - Ek) + Gl * (V - El)))
    return differential_value


def voltage(cur_voltage, t, delta_t, m, n, h):
    diff_equation = lambda y, x: voltage_diff_equation(y, current(x), m, n, h)
    return runge_kutta_method(diff_equation, cur_voltage, t, delta_t)


def runge_kutta_method(diff_equation, y, x, delta_x):
    k1 = diff_equation(y, x)
    k2 = diff_equation(y + k1 * delta_x / 2., x + delta_x / 2.)
    k3 = diff_equation(y + k2 * delta_x / 2., x + delta_x / 2.)
    k4 = diff_equation(y + k3 * delta_x, x + delta_x)

    function_value = V + delta_x * (k1 + 2 * k2 + 2 * k3 + k4) / 6.
    return function_value


if __name__ == "__main__":
    v0 = 0.
    m0 = channel_opening_prob_at_rest(alpha_m, beta_m, v0)
    n0 = channel_opening_prob_at_rest(alpha_n, beta_n, v0)
    h0 = channel_opening_prob_at_rest(alpha_h, beta_h, v0)

    v = v0
    m = m0
    n = n0
    h = h0

    max_t = 100.
    delta_t = 1.
    t_list = np.arange(0., max_t, delta_t)
    v_list = []
    m_list = []
    n_list = []
    h_list = []

    for t in t_list:
        v_list.append(v)
        m_list.append(m)
        n_list.append(n)
        h_list.append(h)

        m = channel_opening_prob(alpha_m, beta_m, v, m, delta_t)
        n = channel_opening_prob(alpha_n, beta_n, v, n, delta_t)
        h = channel_opening_prob(alpha_h, beta_h, v, h, delta_t)
        v = voltage(v, t, delta_t, m, n, h)

    plt.plot(t_list, v_list)
    plt.show()
