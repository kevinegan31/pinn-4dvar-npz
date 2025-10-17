import numpy as np
def npz_nl(x, t, Vm, ks, m, Rm, ivlev, gamma, q):
    x = np.asarray(x)
    dx = np.zeros(x.shape)

    
    dx[1] = (((Vm * x[0]) / (ks + x[0])) * x[1]) - (m * x[1]) - (ivlev * x[1] * Rm * (1 - np.exp(-ivlev * x[1])) * x[2])
    dx[2] = ((1 - gamma) * ivlev * x[1] * Rm * (1 - np.exp(-ivlev * x[1])) * x[2]) - (q * x[2])
    dx[0] = -(((Vm * x[0]) / (ks + x[0])) * x[1]) + (m * x[1]) + (q * x[2]) + (gamma * ivlev * x[1] * Rm * (1 - np.exp(-ivlev * x[1])) * x[2])
    
    return dx
    
def rk4(f, x0, times, tfrc, frc, args):
    """
    Perform the model integration using the Runge-Kutta 4. This
    allows for the function to be forced at given times, ft, by the forcing, f.
    """
    nt = len(times)
    x = np.zeros((nt, len(x0)))
    x[0, :] = x0
    dt = np.zeros(nt)
    dt[1:] = np.diff(times)
    for n in range(1, nt):
        k1 = f(x[n - 1, :], times[n], *args) * dt[n]
        k2 = f(x[n - 1, :] + 0.5 * k1, times[n] + 0.5 * dt[n], *args) * dt[n]
        k3 = f(x[n - 1, :] + 0.5 * k2, times[n] + 0.5 * dt[n], *args) * dt[n]
        k4 = f(x[n - 1, :] + k3, times[n] + dt[n], *args) * dt[n]
        x[n, :] = x[n - 1, :] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        if tfrc is not None:
            fl = np.where(np.logical_and(tfrc >= times[n] - 0.5 * dt[n],
                                         tfrc < times[n] + 0.5 * dt[n]))[0]
            x[n, :] += f[fl, :].sum(axis=0)

    return x
