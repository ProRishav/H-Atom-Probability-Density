import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Optional, Final, Literal
from scipy.special import sph_harm, genlaguerre
import matplotlib.colors as mcolors


class Unassigned:
    ...

class HydrogenAtom:
    _a0: Final[float] = 1.0
    __version__ = "1.0.0"
    def __init__(self) -> None:...
    
    @property
    def a0(self):
        return self._a0
    
    # Radial part of the wave function
    def radial_function(self, n:int, l:int, r:Union[np.ndarray, int, float]) -> Union[np.ndarray, np.float64]:
        if not isinstance(n, int):
            raise TypeError(f"n should be integer. Given:- {n}")
        if n < 1:
            raise ValueError(f"n should not be less than 1. Given:- {n}")
        if not isinstance(l, int):
            raise TypeError(f"l should be integer. Given:- {l}")
        if l >= n:
            raise ValueError(f"l should be less than {n}. Given:- {l}")
        if l < 0:
            raise ValueError(f"l should not be less than 0. Given:- {l}")
        try:
            self.validate_iterable_int_float(r)
        except (ValueError, TypeError) as e:
            print(f"Error: {e}")
        
        rho = 2 * r / (n * self._a0)
        normalization: [int | float] = np.sqrt((2 / (n * self._a0))**3 * math.factorial(n - l - 1)\
            / (2 * n * math.factorial(n + l)))
        laguerre_poly: np.ndarray = genlaguerre(n - l - 1, 2 * l + 1)(rho)
        radial_part: np.ndarray = normalization * np.exp(-rho / 2) * rho**l * laguerre_poly
        return radial_part

    # Angular part of the wave function (Spherical Harmonics)
    def angular_funtion(self, l:int, m:int, theta:Union[np.ndarray, int, float], \
        phi:Union[np.ndarray, int, float]) -> Union[np.array, np.float64]:
        if not isinstance(l, int):
            raise TypeError(f"l should be integer. Given:- {l}")
        if l < 0:
            raise ValueError(f"l should not be less than 1. Given:- {l}")
        if not isinstance(m, int):
            raise TypeError(f"m should be integer. Given:- {m}")
        if abs(m)>l:
            raise ValueError(f"m should be in between {-l-1} and {l+1}. Given:- {m}")
        try:
            self.validate_iterable_int_float(theta)
        except (ValueError, TypeError) as e:
            print(f"Error: {e}")
        try:
            self.validate_iterable_int_float(phi)
        except (ValueError, TypeError) as e:
            print(f"Error: {e}")
        
        return sph_harm(m, l, phi, theta)
    
    # Probability density of the electron
    def probability_density(self, n:int, l:int, m:int, r:Union[np.ndarray, int, float],\
        theta:Union[np.ndarray, int, float], \
            phi:Union[np.ndarray, int, float]) -> Union[np.ndarray, np.float64]:
        if not isinstance(n, int):
            raise TypeError(f"n should be integer. Given:- {n}")
        if n < 1:
            raise ValueError(f"n should not be less than 1. Given:- {n}")
        if not isinstance(l, int):
            raise TypeError(f"l should be integer. Given:- {l}")
        if l >= n:
            raise ValueError(f"l should be less than {n}. Given:- {l}")
        if l < 0:
            raise ValueError(f"l should not be less than 0. Given:- {l}")
        if not isinstance(m, int):
            raise TypeError(f"m should be integer. Given:- {m}")
        if abs(m)>l:
            raise ValueError(f"m should be in between {-l-1} and {l+1}. Given:- {m}")
        try:
            self.validate_iterable_int_float(theta)
        except (ValueError, TypeError) as e:
            print(f"Error: {e}")
        try:
            self.validate_iterable_int_float(phi)
        except (ValueError, TypeError) as e:
            print(f"Error: {e}")
        
        R:[np.array, np.float64] = self.radial_function(n, l, r)
        Y:[np.array, np.float64] = self.angular_funtion(l, m, theta, phi)
        psi:[np.array, np.float64] = R * Y
        return np.abs(psi)**2
    
    # Validate the input
    @staticmethod
    def validate_iterable_int_float(iterable) -> Literal[0, 1]:
        if isinstance(iterable, Union[int, float]): return 1
        if not isinstance(iterable,np.ndarray):
            raise TypeError("Input must be numpy array.")
        if iterable.size == 0:
            raise ValueError("Should not be empty array.")
        if not np.issubdtype(iterable.dtype, np.number):
            raise TypeError("All items in the NumPy array must be of type int or float.")
        return 0
    
def main():
    atom = HydrogenAtom()
    
    # Principal quantum number, azimuthal quantum number, magnetic quantum number
    n, l, m = (4, 1, 1)
    
    r = np.linspace(0, 50*atom._a0, 500)
    theta = np.linspace(0, 2*np.pi, 500)
    R, THETA = np.meshgrid(r, theta)
    density = atom.probability_density(n, l, m, r, THETA, 0)
    X = R * np.sin(THETA)
    Y = R * np.cos(THETA)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1)
    norm = mcolors.Normalize(vmin=0, vmax=1)
    # Change the brightness of the color based on the density (adjust this as needed)
    bright = 5000 * (1 + n**2) / (1 + np.log1p(n))
    
    plt.contourf(X, Y, density*bright, 100, cmap='inferno', norm=norm)
    plt.colorbar(label='Density')
    plt.axis('equal')
    ax.set_facecolor('k')
    plt.show()


if __name__ == "__main__":
    main()
    
    
    
    