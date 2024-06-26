import numpy as np
import matplotlib.pyplot as plt
from numba import njit


def winding_number(j, N=8192, plot=False, **kwargs):
    e = sample_values(j, N)
    if plot:
        plt.plot(e.real, e.imag, color='C0', **kwargs)
        # add arrows at beginning and middle to show direction
        for i in (0, N//2):
            rdir, idir = e.real[i + 1] - e.real[i], e.imag[i + 1] - e.imag[i]
            rdir, idir = rdir / np.hypot(rdir, idir) * 0.01, idir / np.hypot(rdir, idir) * 0.01
            plt.arrow(e.real[i]-rdir, e.imag[i]-idir, 2*rdir, 2*idir,
                        head_width=0.05, head_length=0.2, fc='C0', ec='C0', zorder=10)
        plt.plot([0], [0], marker='o', markersize=5, color="red")
    return winding_number_of_path(e.real, e.imag)


def sample_values(j, N=8192):
    """Evaluate Laurent polynomial j(z) (with equally many positive
    and negative powers) counterclockwise at N evenly spaced roots of
    unity z, wrapping back around to z=1, using FFT"""
    assert N % 2 == 0 and len(j) % 2 == 1
    Tau = len(j) // 2 + 1 # Tau-1 is the maximum pos or neg power in j(z)

    # center j(z) at N/2
    jj = np.zeros(N)
    jj[N//2-Tau+1:N//2+Tau] = j

    # take FFT to evaluate j(z) * z^(N/2) at roots of unity (could exploit conjugate symmetry to halve work)
    e = np.fft.fft(jj)

    # divide by z^(N/2) at same roots, which is alternating 1 and -1, to get j(z)
    alt = np.tile([1, -1], N//2)
    e = e * alt

    # return wrapped back to z=1, reversed to make counterclocwise
    return np.concatenate((e, [e[0]]))[::-1]


@njit
def winding_number_of_path(x, y):
    """Compute winding number around origin of (x,y) coordinates that make closed path by
    counting number of counterclockwise crossings of ray from (0,0) -> (infty,0) on x axis"""
    # ensure closed path!
    assert x[-1] == x[0] and y[-1] == y[0]

    winding_number = 0

    # we iterate through coordinates (x[i], y[i]), where cur_sign is flag for
    # whether current coordinate is above the x axis
    cur_sign = (y[0] >= 0)
    for i in range(1, len(x)):
        if (y[i] >= 0) != cur_sign:
            # if we're here, this means the x axis has been crossed
            # this generally happens rarely, so efficiency no biggie
            cur_sign = (y[i] >= 0)

            # crossing of x axis implies possible crossing of ray (0,0) -> (infty,0)
            # we will evaluate three possible cases to see if this is indeed the case
            if x[i] > 0 and x[i - 1] > 0:
                # case 1: both (x[i-1],y[i-1]) and (x[i],y[i]) on right half-plane, definite crossing
                # increment winding number if counterclockwise (negative to positive y)
                # decrement winding number if clockwise (positive to negative y)
                winding_number += 2 * cur_sign - 1
            elif not (x[i] <= 0 and x[i - 1] <= 0):
                # here we've ruled out case 2: both (x[i-1],y[i-1]) and (x[i],y[i]) in left 
                # half-plane, where there is definitely no crossing

                # thus we're in ambiguous case 3, where points (x[i-1],y[i-1]) and (x[i],y[i]) in
                # different half-planes: here we must analytically check whether we crossed
                # x-axis to the right or the left of the origin
                # [this step is intended to be rare]
                cross_coord = (x[i - 1] * y[i] - x[i] * y[i - 1]) / (y[i] - y[i - 1])
                if cross_coord > 0:
                    winding_number += 2 * cur_sign - 1
    return winding_number
