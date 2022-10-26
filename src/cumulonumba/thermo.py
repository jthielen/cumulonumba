# Copyright (c) 2022 Cumulonumba Developers.
# Distributed under the terms of the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Thermodynamic functions.

Reused with modification from MetPy (Copyright (c) 2008-2020, MetPy Developers) under the
terms of the BSD 3-Clause License (reproduced below).

---

BSD 3-Clause License

Copyright (c) 2008-2020, MetPy Developers
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numba
from numbalsoda import lsoda_sig, lsoda
import numpy as np

from .constants import Cp_d, Rd, Lv, epsilon, kappa, sat_pressure_0c, zero_degc
from .util import _isclose, _steffensens_method


@numba.njit
def dewpoint(vapor_pressure):
    r"""Calculate the ambient dewpoint given the vapor pressure.

    Parameters
    ----------
    vapor_pressure : float
        Water vapor partial pressure, in Pa

    Returns
    -------
    float
        Dewpoint temperature, in K

    Notes
    -----
    This function inverts the [Bolton1980]_ formula for saturation vapor
    pressure to instead calculate the temperature. This yields the following formula for
    dewpoint in degrees Celsius, where :math:`e` is the ambient vapor pressure in millibars:

    .. math:: T = \frac{243.5 \log(e / 6.112)}{17.67 - \log(e / 6.112)}

    .. versionchanged:: 1.0
       Renamed ``e`` parameter to ``vapor_pressure``

    """
    val = np.log(vapor_pressure / sat_pressure_0c)
    return zero_degc + 243.5 * val / (17.67 - val)


@numba.njit
def mixing_ratio(partial_press, total_press):
    r"""Calculate the mixing ratio of a gas.

    This calculates mixing ratio given its partial pressure and the total pressure of
    the air. There are no required units for the input arrays, other than that
    they have the same units.

    Parameters
    ----------
    partial_press : float
        Partial pressure of the constituent gas, in Pa

    total_press : float
        Total air pressure, in Pa

    Returns
    -------
    float
        The (mass) mixing ratio, dimensionless.

    Notes
    -----
    This function is a straightforward implementation of the equation given in many places,
    such as [Hobbs1977]_ pg.73:

    .. math:: r = \epsilon \frac{e}{p - e}

    """
    return epsilon * partial_press / (total_press - partial_press)


@numba.njit
def saturation_mixing_ratio(total_press, temperature):
    r"""Calculate the saturation mixing ratio of water vapor.

    This calculation is given total atmospheric pressure and air temperature.

    Parameters
    ----------
    total_press: float
        Total atmospheric pressure, in Pa

    temperature: float
        Air temperature, in K

    Returns
    -------
    float
        Saturation mixing ratio, dimensionless

    Notes
    -----
    This function is a straightforward implementation of the equation given in many places,
    such as [Hobbs1977]_ pg.73:

    .. math:: r_s = \epsilon \frac{e_s}{p - e_s}

    """
    return mixing_ratio(saturation_vapor_pressure(temperature), total_press)


@numba.njit
def vapor_pressure(pressure, mixing_ratio):
    r"""Calculate water vapor (partial) pressure.

    Given total ``pressure`` and water vapor ``mixing_ratio``, calculates the
    partial pressure of water vapor.

    Parameters
    ----------
    pressure : float
        Total atmospheric pressure, in Pa

    mixing_ratio : float
        Mass mixing ratio, dimensionless.

    Returns
    -------
    float
        Ambient water vapor (partial) pressure, in Pa.

    Notes
    -----
    This function is a straightforward implementation of the equation given in many places,
    such as [Hobbs1977]_ pg.71:

    .. math:: e = p \frac{r}{r + \epsilon}

    """
    return pressure * mixing_ratio / (epsilon + mixing_ratio)


@numba.njit
def saturation_vapor_pressure(temperature):
    r"""Calculate the saturation water vapor (partial) pressure.

    Parameters
    ----------
    temperature : float
        Air temperature, in K

    Returns
    -------
    float
        Saturation water vapor (partial) pressure, in Pa

    Notes
    -----
    Instead of temperature, dewpoint may be used in order to calculate
    the actual (ambient) water vapor (partial) pressure.

    The formula used is that from [Bolton1980]_ for T in degrees Celsius:

    .. math:: 6.112 e^\frac{17.67T}{T + 243.5}

    """
    # Converted from original in terms of C to use kelvin.
    return sat_pressure_0c * np.exp(
        17.67 * (temperature - 273.15) / (temperature - 29.65)
    )


@numba.njit
def _lcl_iter(p, extra):
    p0, w, t = extra
    td = dewpoint(vapor_pressure(p, w))
    p_new = (p0 * (td / t) ** (1. / kappa))
    return p if np.isnan(p_new) else p_new


@numba.njit
def _lcl_scalar(pressure, temperature, dewpoint_value, max_iters=50, eps=1e-5):
    r"""Calculate the lifted condensation level (LCL) from the starting point.

    The starting state for the parcel is defined by `temperature`, `dewpoint`,
    and `pressure`.

    Parameters
    ----------
    pressure : float
        Starting atmospheric pressure, in Pa

    temperature : float
        Starting temperature, in K

    dewpoint_value : float
        Starting dewpoint, in K

    Returns
    -------
    numpy.ndarray
        Length 2 array containing LCL pressure and temperature, in Pa and K, respectively.

    Other Parameters
    ----------------
    max_iters : int, optional
        The maximum number of iterations to use in calculation, defaults to 50.

    eps : float, optional
        The desired relative error in the calculated value, defaults to 1e-5.

    Notes
    -----
    This function is implemented using an iterative approach to solve for the
    LCL. The basic algorithm is:

    1. Find the dewpoint from the LCL pressure and starting mixing ratio
    2. Find the LCL pressure from the starting temperature and dewpoint
    3. Iterate until convergence

    """
    w = mixing_ratio(saturation_vapor_pressure(dewpoint_value), pressure)
    extra = np.array((pressure, w, temperature))
    lcl_p = _steffensens_method(_lcl_iter, pressure, extra, xtol=eps, maxiter=max_iters)

    # isclose needed if surface is LCL due to precision error with np.log in dewpoint.
    # Causes issues with parcel_profile_with_lcl if removed. Issue #1187
    lcl_p = pressure if _isclose(lcl_p, pressure) else lcl_p

    return np.array((lcl_p, dewpoint(vapor_pressure(lcl_p, w))))


@numba.njit
def dry_lapse(pressure, temperature):
    r"""Calculate the temperature at a level assuming only dry processes.

    This function lifts a parcel starting at ``temperature``, conserving
    potential temperature. The starting pressure can be given by ``reference_pressure``.

    Parameters
    ----------
    pressure : float
        Atmospheric pressure level(s) of interest, in Pa

    temperature : float
        Starting temperature, in K

    Returns
    -------
    float
       The parcel's resulting temperature at levels given by ``pressure``, in K`

    """
    reference_pressure = pressure[0]
    return temperature * (pressure / reference_pressure)**kappa


@numba.cfunc(lsoda_sig, nopython=True)
def _moist_lapse_rhs(pdiff, t, dt, x):
    p = x[8] + x[9] * pdiff
    # numba vs. c type weirdness with constants; see data in _moist_lapse_1d for x values.
    partial_press = x[0] * np.exp(x[1] * (t[0] - x[2]) / (t[0] - x[3]))
    rs = x[4] * partial_press / (p - partial_press)
    frac = (x[5] * t[0] + x[6] * rs) / (x[7] + (x[6] * x[6] * rs * x[4] / (x[5] * t[0] * t[0])))
    dt[0] = x[9] * frac / p


_moist_lapse_rhs_address = _moist_lapse_rhs.address


@numba.njit
def _moist_lapse_1d(pressure, temperature, reference_pressure):
    r"""Calculate the temperature at a level assuming liquid saturation processes.

    This function lifts a parcel starting at `temperature`. The starting pressure can
    be given by `reference_pressure`. Essentially, this function is calculating moist
    pseudo-adiabats.

    Parameters
    ----------
    pressure : float
        Atmospheric pressure level(s) of interest, in Pa.

    temperature : float
        Starting temperature, in K.

    reference_pressure : float
        Reference pressure, in Pa.

    Returns
    -------
    float
       The resulting parcel temperature at levels given by `pressure`, in K.

    Notes
    -----
    This function is implemented by integrating the following differential
    equation:

    .. math:: \frac{dT}{dP} = \frac{1}{P} \frac{R_d T + L_v r_s}
                                {C_{pd} + \frac{L_v^2 r_s \epsilon}{R_d T^2}}

    This equation comes from [Bakhshaii2013]_.

    """
    temperature = np.array([temperature])
    pressure = np.atleast_1d(pressure)

    if np.isnan(reference_pressure) or np.all(np.isnan(temperature)):
        return np.full_like(pressure, np.nan)

    pres_decreasing = (pressure[0] > pressure[-1])
    if pres_decreasing:
        # Everything is easier if pressures are in increasing order
        pressure = pressure[::-1]

    # lsoda args
    funcptr = _moist_lapse_rhs_address
    u0 = temperature
    # try using data to pass needed constants
    data_fixed = np.array([sat_pressure_0c, 17.67, 273.15, 29.65, epsilon, Rd, Lv, Cp_d])  

    # Need to handle close points to avoid an error in the solver
    close = _isclose(pressure, reference_pressure)
    if np.any(close):
        ret = np.full((np.sum(close), 1), temperature[0])
    else:
        ret = np.empty((0, 1), dtype=temperature.dtype)

    # Do we have any points above the reference pressure
    points_above = (pressure < reference_pressure) & ~close
    if np.any(points_above):
        # Integrate upward--need to flip so values are properly ordered from ref to min
        press_diff = np.append(0, reference_pressure - pressure[points_above][::-1])

        # Flip on exit so t values correspond to increasing pressure
        data = np.concatenate((data_fixed, np.array([reference_pressure, -1.])))
        trace, success = lsoda(funcptr, u0, press_diff, data=data, atol=1e-7, rtol=1.5e-8)
        ret = np.concatenate((trace[:0:-1], ret), axis=0)
      
    # Do we have any points below the reference pressure
    points_below = ~points_above & ~close
    if np.any(points_below):
        # Integrate downward
        press_diff = np.append(0, pressure[points_below] - reference_pressure)
        data = np.concatenate((data_fixed, np.array([reference_pressure, 1.])))
        trace, success = lsoda(funcptr, u0, press_diff, data=data, atol=1e-7, rtol=1.5e-8)
        ret = np.concatenate((ret, trace[1:]), axis=0)

    if pres_decreasing:
        ret = ret[::-1]

    return ret[..., 0]
