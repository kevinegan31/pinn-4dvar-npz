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

# RK4 for a single set of initial conditions
def run_rk4_for_initial_conditions(N0, P0, Z0, t, phi):
    # Define the initial conditions as a list
    x0 = [N0, P0, Z0]
    # Use the rk4 method to expand the system over time
    xt = rk4(npz_nl, x0, t, None, 0, phi)
    return xt



def npz_tl(tl_x, x, t, Vm, ks, m, Rm, ivlev, gamma, q):
    x = np.asarray(x)  # Base state (N, P, Z)
    tl_dx = np.zeros(x.shape)  # Initialize perturbations derivatives (delta N, delta P, delta Z)

    # Derivative of P
    tl_dx[1] = (
        ((Vm * x[0]) / (ks + x[0])) * tl_x[1] +
        ((Vm * ks * x[1]) / ((ks + x[0])**2)) * tl_x[0] - 
        m * tl_x[1] -
        (ivlev * Rm * x[2] * ((1 - np.exp(-ivlev * x[1])) + (ivlev * np.exp(-ivlev * x[1]) * x[1]))) * tl_x[1] -
        (ivlev * Rm * x[1] * (1 - np.exp(-ivlev * x[1]))) * tl_x[2])

    # Derivative of Z
    tl_dx[2] = (
        (ivlev * (1 - gamma) * Rm * x[1] * (1 - np.exp(-ivlev * x[1]))) * tl_x[2] +
        (ivlev * (1 - gamma) * Rm * x[2] * (1 - np.exp(-ivlev * x[1]) + ivlev * x[1] * np.exp(-ivlev * x[1]))) * tl_x[1] -
        q * tl_x[2]  # Zooplankton mortality
    )

    # Derivative of N
    tl_dx[0] = (
        -(((Vm * x[0]) / (ks + x[0])) * tl_x[1] +
        ((Vm * ks * x[1]) / ((ks + x[0])**2)) * tl_x[0]) + 
        m * tl_x[1] +
        q * tl_x[2] +
        (gamma * ivlev * Rm * x[2] * (1 - np.exp(-ivlev * x[1]) + ivlev * np.exp(-ivlev * x[1]) * x[1])) * tl_x[1] +
        (gamma * ivlev * Rm * x[1] * (1 - np.exp(-ivlev * x[1])) * tl_x[2])
    )
    return tl_dx

def rk4_tl(f, tl_f, x, tl_x0, times, tfrc, frc, args):
    """
    Perform the gradient integration using the tangent-linear of Runge-Kutta 4. This
    allows for the function to be forced at given times, ft, by the forcing, f.
    """
    nt = len(times)
    tl_x = np.zeros((nt, len(tl_x0)))
    tl_x[0, :] = tl_x0
    dt = np.zeros(nt)
    dt[1:] = np.diff(times)
    for n in range(1, nt):
        k1 = f(x[n - 1, :], times[n], *args) * dt[n]
        k2 = f(x[n - 1, :] + 0.5 * k1, times[n] + 0.5 * dt[n], *args) * dt[n]
        k3 = f(x[n - 1, :] + 0.5 * k2, times[n] + 0.5 * dt[n], *args) * dt[n]
        k4 = f(x[n - 1, :] + k3, times[n] + dt[n], *args) * dt[n]

        tl_k1 = tl_f(tl_x[n - 1, :], x[n - 1, :], times[n], *args) * dt[n]
        tl_k2 = tl_f(tl_x[n - 1, :] + 0.5 * tl_k1,
                     x[n - 1, :] + 0.5 * k1, times[n], * args) * dt[n]
        tl_k3 = tl_f(tl_x[n - 1, :] + 0.5 * tl_k2,
                     x[n - 1, :] + 0.5 * k2, times[n], * args) * dt[n]
        tl_k4 = tl_f(tl_x[n - 1, :] + tl_k3, x[n - 1, :] +
                     k3, times[n] + dt, * args) * dt[n]
        tl_x[n, :] = tl_x[n - 1, :] + \
            (tl_k1 + 2 * tl_k2 + 2 * tl_k3 + tl_k4) / 6

        # Apply the forcing if it is at a forcing time
        if tfrc is not None:
            fl = np.where(np.logical_and(tfrc >= times[n] - 0.5 * dt[n],
                                         tfrc < times[n] + 0.5 * dt[n]))[0]
            tl_x[n, :] += frc[fl, :].sum(axis=0)

    return tl_x
    
# Adjoint Model Code
def npz_ad(ad_x, x, t, Vm, ks, m, Rm, ivlev, gamma, q):
    # Initialize the adjoint derivative array
    ad_dx = np.zeros(x.shape)

    # Adjoint equation for N
    ad_dx[1] -= ((Vm * x[0]) / (ks + x[0])) * ad_x[0]  # Nutrient uptake (affecting P)
    ad_dx[0] -= ((Vm * ks * x[1]) / ((ks + x[0])**2)) * ad_x[0]  # Nutrient limitation (affecting N)
    ad_dx[1] += m * ad_x[0]  # Phytoplankton mortality
    ad_dx[2] += q * ad_x[0]  # Zooplankton mortality

    ad_dx[1] += (gamma * ivlev * Rm * x[2] * (1 - np.exp(-ivlev * x[1]) + ivlev * np.exp(-ivlev * x[1]) * x[1])) * ad_x[0]  # Grazing on P
    ad_dx[2] += (gamma * ivlev * Rm * x[1] * (1 - np.exp(-ivlev * x[1]))) * ad_x[0]  # Grazing on Z
    
    # Adjoint equation for Z
    ad_dx[2] += (ivlev * (1 - gamma) * Rm * x[1] * (1 - np.exp(-ivlev * x[1]))) * ad_x[2]
    ad_dx[1] += (ivlev * (1 - gamma) * Rm * x[2] * (1 - np.exp(-ivlev * x[1]) + ivlev * x[1] * np.exp(-ivlev * x[1]))) * ad_x[2]
    ad_dx[2] -= q * ad_x[2]  # Zooplankton mortality

    # Adjoint equation for P
    ad_dx[1] += ((Vm * x[0]) / (ks + x[0])) * ad_x[1]  # Nutrient uptake (affecting P)
    ad_dx[0] += ((Vm * ks * x[1]) / ((ks + x[0])**2)) * ad_x[1]  # Nutrient limitation (affecting N)
    ad_dx[1] -= m * ad_x[1]  # Phytoplankton mortality

    ad_dx[1] -= (ivlev * Rm * x[2] * ((1 - np.exp(-ivlev * x[1])) + ivlev * np.exp(-ivlev * x[1]) * x[1])) * ad_x[1]  # P grazing by Z
    ad_dx[2] -= (ivlev * Rm * x[1] * (1 - np.exp(-ivlev * x[1]))) * ad_x[1]  # Z grazing on P



    return ad_dx

def rk4_ad(f, ad_f, x, ad_x0, times, tfrc, frc, args):
    """
    Perform the adjoint integration using the adjoing of Runge-Kutta 4. This
    allows for the function to be forced at given times, ft, by the forcing, f.
    """
    nt = len(times)
    ad_x = np.zeros((nt, len(x0)))
    ad_x[-1, :] = ad_x0
    dt = np.zeros(nt)
    dt[1:] = np.diff(times)
    for n in range(nt - 1, 0, -1):
        k1 = f(x[n - 1, :], times[n], *args) * dt[n]
        k2 = f(x[n - 1, :] + 0.5 * k1, times[n] + 0.5 * dt[n], *args) * dt[n]
        k3 = f(x[n - 1, :] + 0.5 * k2, times[n] + 0.5 * dt[n], *args) * dt[n]
        k4 = f(x[n - 1, :] + k3, times[n] + dt[n], *args) * dt[n]

        # Check if we are at a forcing-time. If so, add the forcing to the
        # solution.
        # if tfrc is not None:
        #     fl = np.where(np.logical_and(tfrc >= times[n] - 0.5 * dt[n],
        #                                  tfrc < times[n] + 0.5 * dt[n]))[0]
        #     ad_x[n - 1, :] += frc[fl, :].sum(axis=0)
        if tfrc is not None:
            ad_x[n - 1, :] += frc[n - 1, :]

        # final-step
        ad_k1 = ad_x[n, :] / 6
        ad_k2 = 2 * ad_x[n, :] / 6
        ad_k3 = 2 * ad_x[n, :] / 6
        ad_k4 = ad_x[n, :] / 6
        ad_x[n - 1, :] += ad_x[n, :]

        # k4-step
        ad_x[n - 1, :] += ad_f(ad_k4 * dt[n], x[n - 1, :] +
                               k3, times[n] + dt[n], *args)
        ad_k3 += ad_f(ad_k4 * dt[n], x[n - 1, :] + k3, times[n] + dt[n], *args)

        # k3-step
        ad_x[n - 1, :] += ad_f(ad_k3 * dt[n], x[n - 1, :] +
                               0.5 * k2, times[n] + 0.5 * dt[n], *args)
        ad_k2 += ad_f(ad_k3 * 0.5 * dt[n], x[n - 1] +
                      0.5 * k2, times[n] + 0.5 * dt[n], *args)

        # k2-step
        ad_x[n - 1, :] += ad_f(ad_k2 * dt[n], x[n - 1, :] +
                               0.5 * k1, times[n] + 0.5 * dt[n], *args)
        ad_k1 += ad_f(ad_k2 * 0.5 * dt[n], x[n - 1, :] +
                      0.5 * k1, times[n] + 0.5 * dt[n], *args)

        # k1-step
        ad_x[n - 1, :] += ad_f(ad_k1 * dt[n], x[n - 1, :], times[n], *args)

        # tl_k1 = tl_f(tl_x[n - 1, :], x[n - 1, :], times[n], *args) * dt[n]
        # tl_k2 = tl_f(tl_x[n - 1, :] + 0.5 * tl_k1,
        #              x[n - 1, :] + 0.5 * k1, times[n], * args) * dt[n]
        # tl_k3 = tl_f(tl_x[n - 1, :] + 0.5 * tl_k2,
        #              x[n - 1, :] + 0.5 * k2, times[n], * args) * dt[n]
        # tl_k4 = tl_f(tl_x[n - 1, :] + tl_k3, x[n - 1, :] +
        #              k3, times[n] + dt, * args) * dt[n]
        # tl_x[n, :] = tl_x[n - 1, :] + (tl_k1 + 2 * tl_k2 + 2 * tl_k3 + tl_k4) / 6

    return ad_x
