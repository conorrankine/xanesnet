"""
XANESNET-REDUX
Copyright (C) 2025  Conor D. Rankine

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software 
Foundation, either Version 3 of the License, or (at your option) any later 
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with 
this program.  If not, see <https://www.gnu.org/licenses/>.
"""

###############################################################################
############################### LIBRARY IMPORTS ###############################
###############################################################################

import matplotlib.pyplot as plt
import numpy as np
import copy
from pathlib import Path
from typing import Union, TextIO
from typing_extensions import Self
from .templates import BaseTransformer

###############################################################################
################################## CLASSES ####################################
###############################################################################

class XANES():

    def __init__(
        self,
        energy: np.ndarray,
        absorption: np.ndarray,
        absorption_edge: float = None
    ):
        """
        Args:
            energy (np.ndarray): Array of energy values (in eV) defining the
                XANES spectrum.
            absorption (np.ndarray): Array of absorption intensity values
                defining the XANES spectrum.
            absorption_edge (float, optional): X-ray absorption edge energy
                (in eV). If None, the X-ray absorption edge is estimated as
                coincident with the maximum derivative of the absorption
                intensity. Defaults to None.

        Raises:
            ValueError: If `energy` and `absorption` are not the same length.
        """
        
        if len(energy) == len(absorption):
            self._energy = energy
            self._absorption = absorption
        else:
            raise ValueError(
                'the energy and absorption intensity arrays are not the same '
                f'length ({len(energy)} vs. {len(absorption)})'
            )

        if absorption_edge is not None:
            self._absorption_edge = absorption_edge
        else:
            self._absorption_edge = self.estimate_absorption_edge()

        self._energy_rel = self._energy - self._absorption_edge

    def estimate_absorption_edge(
        self
    ) -> float:
        """
        Estimates the energy of the X-ray absorption edge (in eV) as the
        energy at which the derivative of the absorption intensity is largest.

        Returns:
            float: Estimated energy of the X-ray absorption edge (in eV).
        """

        return self._energy[np.argmax(np.gradient(self._absorption))]
    
    def shift(
        self,
        shift: float = 0.0,
        inplace: bool = True
    ) -> Self:
        """
        Shifts the XANES spectrum in energy by `shift` eV, updating the
        `energy` and `absorption_edge` properties.

        Args:
            shift (float, optional): Shift in energy (in eV). Defaults to 0.0.
            inplace (bool, optional): If `True`, the transformation is carried
                out in-place; if `False`, the transformation is carried out on
                a copy and the copy is returned. Defaults to `True`.

        Returns:
            Self: Self if `inplace` is `True`, else a new `XANES` instance with
                the transformation applied.
        """
        
        result = self if inplace else copy.deepcopy(self)

        result._energy += shift
        result._absorption_edge += shift

        return result
    
    def scale(
        self,
        scale: float = 1.0,
        inplace: bool = True
    ) -> Self:
        """
        Scales the XANES spectrum in (absorption) intensity by a factor of
        `scale`, updating the `absorption` property.

        Args:
            scale (float, optional): Scale. Defaults to 1.0.
            inplace (bool, optional): If `True`, the transformation is carried
                out in-place; if `False`, the transformation is carried out on
                a copy and the copy is returned. Defaults to `True`.

        Raises:
            ValueError: If `scale` <= 0.0.

        Returns:
            Self: Self if `inplace` is `True`, else a new `XANES` instance with
                the transformation applied.
        """

        if scale <= 0.0:
            raise ValueError(
                f'scaling factors should be > 0.0; got {scale} (<= 0.0)'
            )

        result = self if inplace else copy.deepcopy(self)

        result._absorption *= scale

        return result
    
    def interp(
        self,
        energy_min: float,
        energy_max: float,
        n_bins: int,
        inplace: bool = True
    ) -> Self:
        """
        Resamples the XANES spectrum via linear interpolation between the
        bounds `energy_min` and `energy_max` at `n_bins` points.

        Args:
            energy_min (float): Minimum energy bound (in eV) relative to the
                X-ray absorption edge.
            energy_max (float): Maximum energy bound (in eV) relative to the
                X-ray absorption edge.
            n_bins (int): Number of energy gridpoints to resample the XANES
                spectrum over.
            inplace (bool, optional): If `True`, the transformation is carried
                out in-place; if `False`, the transformation is carried out on
                a copy and the copy is returned. Defaults to `True`.

        Raises:
            ValueError: If `energy_min` >= `energy_max`.
            ValueError: If `n_bins` <= 0.

        Returns:
            Self: Self if `inplace` is `True`, else a new `XANES` instance with
                the transformation applied.
        """

        if energy_min >= energy_max:
            raise ValueError(
                f'the minimum energy bound ({energy_min} eV) can\'t be '
                f'greater than the maximum energy bound ({energy_max} eV)'
            )
        
        if n_bins <= 0:
            raise ValueError(
                f'the number of discrete energy bins ({n_bins}) can\'t be '
                'fewer than 1'
            )
            
        result = self if inplace else copy.deepcopy(self)

        interp_energy_rel = np.linspace(
            energy_min, energy_max, n_bins
        )
        interp_energy = interp_energy_rel + result._absorption_edge
        interp_absorption = np.interp(
            interp_energy, result._energy, result._absorption
        )

        result._energy = interp_energy
        result._energy_rel = interp_energy_rel
        result._absorption = interp_absorption

        return result

    def normalise(
        self,
        fit_limits: tuple = (100.0, 400.0),
        flatten: bool = True,
        inplace: bool = True
    ) -> Self:
        """
        Normalises the XANES spectrum using the 'edge-step' approach, i.e., by
        fitting a 2nd-order (quadratic) polynomial to (part of) the post-edge
        (where `energy` >= `absorption_energy`), determining the 'edge step',
        `fit(absorption_energy)`, and normalising the absorption intensity
        (`absorption`) by dividing through by this value.
         
        Optionally, the XANES spectrum can be flattened post-normalisation;
        the post-edge can be levelled off to ca. 1.0 by adding (1.0 - 
        `fit(absorption_energy)`) to the absorption intensity (`absorption`)
        where `energy` >= `absorption_energy`.

        Args:
            fit_limits (tuple, optional): Limits (lower and upper; in eV
                relative to the X-ray absorption edge) that bound the energy
                window over which the 2nd-order (quadratic) polynomial is fit.
                Defaults to (100.0, 400.0).
            flatten (bool, optional): Toggles flattening of the post-edge by
                adding (1.0 - `fit(absorption_energy)`) to the absorption
                intensity (`absorption`) where `energy` >= `absorption_energy`.
                Defaults to True.
            inplace (bool, optional): If `True`, the transformation is carried
                out in-place; if `False`, the transformation is carried out on
                a copy and the copy is returned. Defaults to `True`.

        Returns:
            Self: Self if `inplace` is `True`, else a new `XANES` instance with
                the transformation applied.
        """

        result = self if inplace else copy.deepcopy(self)

        min_, max_ = fit_limits

        fit_window = (
            (result._energy_rel >= min_) & (result._energy_rel <= max_)
        )

        fit = np.polynomial.Polynomial.fit(
            result._energy[fit_window],
            result._absorption[fit_window],
            deg = 2
        )

        result._absorption /= fit(result._absorption_edge)

        if flatten:
            postedge_filter = result._energy >= result._absorption_edge
            result._absorption[postedge_filter] += (
                1.0 - (fit(self._energy)[postedge_filter] / 
                    fit(self._absorption_edge))
            )

        return result

    def convolve(
        self,
        conv_params: dict = None,
        inplace: bool = True
    ) -> Self:
        """
        Convolves the XANES spectrum with a fixed-width or energy-dependent
        (variable-width) Lorentzian function where the widths are
        derived from a convolution model (e.g. Seah-Dench; arctangent; etc.).

        XANES spectra are projected onto an auxilliary energy grid via linear
        interpolation pre-convolution; the spacing of the energy gridpoints
        on the auxilliary energy grid are equal to the smallest spacing of the
        energy gridpoints on the original energy grid.

        Args:
            conv_params (dict, optional): Dictionary of convolution parameters,
                including the convolution type (`conv_type`). If these keys
                are *not* set, `conv_type` defaults to 'fixed_width', `width`
                defaults to 1.0 eV, and `ef` (the Fermi energy relative to the
                X-ray absorption edge) defaults to -1.0 eV. Defaults to None.
            inplace (bool, optional): If `True`, the transformation is carried
                out in-place; if `False`, the transformation is carried out on
                a copy and the copy is returned. Defaults to `True`.

        Returns:
            Self: Self if `inplace` is `True`, else a new `XANES` instance with
                the transformation applied.
        """

        result = self if inplace else copy.deepcopy(self)

        conv_params = _set_default_conv_params(conv_params)

        delta = np.min(np.diff(result._energy_rel))

        pad = delta * int((50.0 * conv_params["width"]) / delta)

        energy_aux = np.linspace(
            np.min(result._energy_rel) - pad,
            np.max(result._energy_rel) + pad,
            int((np.ptp(result._energy_rel) + (2.0 * pad)) / delta) + 1
        )

        width = _get_conv_width(
            energy_aux,
            conv_params = conv_params
        )

        preedge_filter = (
            result._energy < (result._absorption_edge + conv_params["ef"])
        )

        # remove cross-sectional contributions to the absorption intensity
        # (`absorption`) below the Fermi energy (`ef`)
        result._absorption[preedge_filter] = 0.0

        # project the absorption intensity (`absorption`) onto the auxilliary
        # energy grid (`e_aux`)
        absorption_aux = np.interp(
            energy_aux, result._energy_rel, result._absorption
        )

        # create the convolutional filter (`conv_filter`)
        conv_filter = _lorentzian(
            *np.meshgrid(energy_aux, energy_aux), width
        )

        # convolve the absorption intensity on the auxilliary energy grid
        # (`absorption_aux`) with the convolution filter (`conv_filter`)
        absorption_aux = np.sum(
            conv_filter * absorption_aux, axis = 1
        )

        # project the absorption intensity on the auxilliary energy grid
        # (`absorption_aux`) onto the original energy grid (`energy`)
        result._absorption = np.interp(
            result._energy_rel, energy_aux, absorption_aux
        )

        return result
    
    def plot(
        self,
        ax: plt.Axes = None,
        figsize: tuple[float, float] = (6.0, 4.0),
        xlabel: str = 'Energy / eV',
        ylabel: str = 'Absorption Intensity',
        title: str = None,
        grid: bool = True,
        **kwargs
    ) -> plt.Axes:
        """
        Plots the XANES spectrum.

        Args:
            ax (plt.Axes, optional): Matplotlib axes (plt.Axes instance) to
                plot the XANES spectrum on. Defaults to None.
            figsize (tuple[float, float], optional): Matplotlib figure size
                [(width, height); inches]; only used if `ax` is None and a new
                'fig/ax' pair is instantiated. Defaults to (6.0, 4.0).
            xlabel (str, optional): X axis label. Defaults to 'Energy / eV'.
            ylabel (str, optional): Y axis label. Defaults to 'Absorption'.
            title (str, optional): Plot title. Defaults to None.
            grid (bool, optional): Toggles the visibility of the axes grid.
                Defaults to True.

        Returns:
            plt.Axes: Matplotlib axes containing the XANES spectrum plot.
        """
        
        if ax is None:
            fig, ax = plt.subplots(figsize = figsize)

        ax.plot(
            self._energy,
            self._absorption,
            **kwargs
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)

        ax.grid(grid)

        plt.tight_layout()

        return ax

    @property
    def energy(
        self
    ) -> np.ndarray:
        """
        Returns:
            np.ndarray: Array of energy values (in eV) defining the XANES
                spectrum.
        """
        
        return self._energy
    
    @property
    def energy_rel(
        self
    ) -> np.ndarray:
        """
        Returns:
            np.ndarray: Array of energy values (in eV relative to the X-ray
                absorption edge) defining the XANES spectrum.
        """
        
        return self._energy_rel

    @property
    def absorption(
        self
    ) -> np.ndarray:
        """
        Returns:
            np.ndarray: Array of absorption intensity values defining the
                XANES spectrum.
        """
        
        return self._absorption

    @property
    def absorption_edge(
        self
    ) -> float:
        """
        Returns:
            float: X-ray absorption edge (in eV).
        """

        return self._absorption_edge

class XANESSpectrumTransformer(BaseTransformer):
    """
    A class for carrying out sequential preprocessing transforms, e.g.,
    shifting, slicing, scaling, interpolating, and convolution, on XANES
    spectra; to be used as part of a data loading/preproccesing pipeline,
    e.g., in `xanesnet/datasets.py`, and to provide a common interface
    matching the `Descriptor` class in `xanesnet/descriptors/` (e.g., by
    exposing the `.transform()` and `.get_len()` methods).
    """

    def __init__(
        self,
        energy_min: float = -30.0,
        energy_max: float = 120.0,
        n_bins: int = 150,
        normalise: bool = True,
        conv: bool = True,
        conv_params: dict = None
    ):
        """
        Args:
            energy_min (float, optional): Minimum energy bound (in eV) for the
                spectral window/slice relative to the X-ray absorption edge.
                Defaults to -30.0.
            energy_max (float, optional): Maximum energy bound (in eV) for the
                spectral window/slice relative to the X-ray absorption edge.
                Defaults to 120.0.
            n_bins (int, optional): Number of discrete energy bins for the
                spectral window/slice. Defaults to 150.
            normalise (bool, optional): Toggles spectrum normalisation using
                the 'edge-step' approach. Defaults to `True`.
            conv (str, optional): Toggles spectrum convolution using (a) fixed-
                or variable-width Lorentzian(s). Defaults to `True`.
            conv_params (dict, optional): Dictionary of convolution parameters,
                including the convolution type (`conv_type`). Defaults to
                None.

        Raises:
            ValueError: If `energy_min` >= `energy_max`.
            ValueError: If `n_bins` <= 0.
        
        """
        
        if energy_min < energy_max:
            self._energy_min = energy_min
            self._energy_max = energy_max
        else:
            raise ValueError(
                f'the minimum energy bound ({energy_min} eV) can\'t be '
                f'greater than the maximum energy bound ({energy_max} eV)'
            )
        
        if n_bins > 0:
            self._n_bins = n_bins
        else:
            raise ValueError(
                'the number of discrete energy bins for the spectral window '
                'can\'t be fewer than 1'
            )
        
        self._energy_aux = np.linspace(
            self._energy_min, self._energy_max, self._n_bins
        )

        self.normalise = normalise

        self.conv = conv
        self.conv_params = conv_params if conv else None

    def transform(
        self,
        spectrum: 'XANES'
    ) -> np.ndarray:
        """
        Transforms a XANES spectrum; carries out sequential preprocessing
        transforms, e.g., shifting, slicing, scaling, interpolating, and
        convolution on a XANES spectrum and returns the transformed spectral
        intensity as a Numpy array (`np.ndarray` object).

        Args:
            spectrum (XANES): XANES spectrum.

        Returns:
            np.ndarray: Transformed XANES spectrum.
        """

        if self.conv:
            spectrum.convolve(conv_params = self.conv_params)

        if self.normalise:
            spectrum.normalise()

        spectrum_ = np.interp(
            self._energy_aux,
            spectrum.energy - spectrum.absorption_edge,
            spectrum.absorption
        )
        
        return spectrum_
    
    @property
    def energy_grid(
        self
    ) -> np.ndarray:
        """
        Returns:
            np.ndarray: Discrete energy bins.
        """
        
        return self._e_aux

    @property
    def size(
        self
    ) -> int:
        """
        Returns:
            int: Number of discrete energy bins.
        """

        return self._n_bins

###############################################################################
################################## FUNCTIONS ##################################
###############################################################################

def _lorentzian(
    x: np.ndarray,
    x0: float,
    width: float
) -> np.ndarray:
    """
    Returns a Lorentzian function defined over `x`.

    Args:
        x (np.ndarray): Array of `x` values at which the Lorentzian function
            is evaluated.
        x0 (float): Peak, or center, of the Lorentzian function.
        width (float): Width (full-width-at-half-maximum; FWHM) of the
            Lorentzian function.

    Returns:
        np.ndarray: Array of `y` values corresponding to the evaluation of
            the Lorentzian function at each value of `x`.
    """
    
    return width * (0.5 / ((x - x0)**2 + (0.5 * width)**2))

def _gaussian(
    x: np.ndarray,
    x0: float,
    sigma: float
) -> np.ndarray:
    """
    Returns a Gaussian function defined over `x`.

    Args:
        x (np.ndarray): Array of `x` values at which the Gaussian function
            is evaluated.
        x0 (float): Peak, or center, of the Gaussian function.
        sigma (float): Width (standard deviation; sigma) of the Gaussian
            function.

    Returns:
        np.ndarray: Array of `y` values corresponding to the evaluation of
            the Gaussian function at each value of `x`.
    """

    return np.exp(-0.5 * ((x - x0) / sigma)**2)

def _calc_seah_dench_conv_width(
    energy_rel: np.ndarray,
    width: float = 1.0,
    width_max: float = 15.0,
    ef: float = -5.0,
    a: float = 1.0
) -> np.ndarray:
    """
    Calculates the widths of the energy-dependent Lorentzians under the Seah-
    Dench convolution model.

    Args:
        energy_rel (np.ndarray): Energies (in eV) relative to the X-ray
            absorption edge.
        width (float, optional): Initial Lorentzian width (in eV). Defaults to
            2.0 (eV).
        width_max (float, optional): Final/maximum Lorentzian width (in eV).
            Defaults to 15.0 (eV).
        ef (float, optional): Fermi energy (in eV) relative to the X-ray
            absorption edge. Defaults to -5.0 (eV).
        a (float, optional): Seah-Dench (pre-)factor. Defaults to 1.0.
        
    Returns:
        np.ndarray: Array of values corresponding to the widths of the energy-
            dependent Lorentzians under the Seah-Dench convolution model.
    """

    e_ = (energy_rel - ef)

    return width + (
        (a * width_max * e_) / (width_max + (a * e_))
    )

def _calc_arctangent_conv_width(
    energy_rel: np.ndarray,
    width: float = 1.0,
    width_max: float = 15.0,
    ef: float = -5.0,
    ec: float = 30.0,
    el: float = 30.0
) -> np.ndarray:
    """
    Calculates the widths of the energy-dependent Lorentzians under the
    arctangent convolution model.

    Args:
        energy_rel (np.ndarray): Energies (in eV) relative to the X-ray
            absorption edge.
        width (float, optional): Initial Lorentzian width (in eV). Defaults to
            2.0 (eV).
        width_max (float, optional): Final/maximum Lorentzian width (in eV).
            Defaults to 15.0 (eV).
        ef (float, optional): Fermi energy (in eV) relative to the X-ray
            absorption edge. Defaults to -5.0 (eV).
        ec (float, optional): Center of the arctangent smoothing function (in
            eV) relative to the X-ray absorption edge. Defaults to 30.0 (eV).
        el (float, optional): Width of the arctangent smoothing function (in
            eV). Defaults to 30.0 (eV).

    Returns:
        np.ndarray: Array of values corresponding to the widths of the energy-
            dependent Lorentzians under the arctangent convolution model.
    """
   
    e_ = (energy_rel - ef) / ec
    with np.errstate(divide = 'ignore'):
        arctan = (np.pi / 3.0) * (width_max / el) * (e_ - (1.0 / e_**2))

    return width + (
        width_max * ((1.0 / 2.0) + (1.0 / np.pi) * np.arctan(arctan))
    )

def _set_default_conv_params(
    conv_params: dict
) -> dict:
    """
    Sets defaults for a dictionary of convolutional parameters; if unset,
        these defaults are:

        'conv_type' (str) = 'fixed_width'
        'width' (float) = 1.0
        'ef' (float) = -1.0

    Args:
        conv_params (dict): Dictionary of convolutional parameters. If None,
            an empty dictionary is instantiated and set with defaults.

    Returns:
        dict: Dictionary of convolutional parameters with defaults set.
    """
    
    if conv_params is None:
        conv_params = {}

    defaults = {
        'conv_type': 'fixed_width',
        'width': 1.0,
        'ef': -1.0
    }

    for key, val in defaults.items():
        conv_params.setdefault(key, val)

    return conv_params    

def _get_conv_width(
    energy_rel: np.ndarray,
    conv_params: dict
) -> Union[float, np.ndarray]:
    """
    Returns the width(s) of the fixed-width or energy-dependent Lorentzian(s)
    under the specified type of convolutional model.

    Args:
        energy_rel (np.ndarray): Energies (in eV) relative to the X-ray
            absorption edge.
        conv_params (dict): Dictionary of convolution parameters,
            including the convolution type (`conv_type`). Defaults to None.

    Raises:
        ValueError: If `conv_type` is not one of 'fixed_width', 'seah_dench',
            or 'arctangent'.

    Returns:
        Union[float, np.ndarray]: Width of the Lorentzian if the convolutional
            model is 'fixed width', else an array of values corresponding to
            the widths of the energy-dependent Lorentzians under the specified
            type of convolutional model if the convolutional model is, e.g.,
            'seah_dench' or 'arctangent'.
    """
    
    params = {
        key: val for key, val in conv_params.items() if key != 'conv_type'
    }

    if conv_params["conv_type"] == 'fixed_width':
            return conv_params['width']
    elif conv_params["conv_type"] == 'seah_dench':
        return _calc_seah_dench_conv_width(energy_rel, **params)
    elif conv_params["conv_type"] == 'arctangent':
        return _calc_arctangent_conv_width(energy_rel, **params)
    else:
        raise ValueError(
            f'convolution type {conv_params["conv_type"]} is not a '
            'recognised/supported convolution type; try, e.g., '
            '\'fixed_width\', \'seah_dench\', or \'arctangent\''                
        )

def get_spectrum_transformer(
    transformer_type: str,
    params: dict = None
) -> 'XANESSpectrumTransformer':
    """
    Returns a spectrum transformer instance of the specified type, optionally
    initialised with a set of parameters that can be passed through to the
    constructor function of the spectrum transformer to override the defaults.

    Args:
        transformer_type (str): Transformer type, e.g., 'xanes' (X-ray
            absorption near-edge spectrum); etc.
        params (dict, optional): Parameters passed through to the constructor
            function of the transformer. Defaults to None.

    Raises:
        ValueError: If `transformer_type` is not an available/valid
            transformer.

    Returns:
        XANESSpectrumTransformer: Transformer.
    """
    
    if params is None:
        params = {}

    transformers = {
        'xanes': XANESSpectrumTransformer
    }

    try:
        return transformers.get(transformer_type)(**params)
    except KeyError:
        raise ValueError(
            f'\'{transformer_type}\' is not an available/valid transformer'
        ) from None

def read(
    filepath: Union[str, Path],
    format: str = None
) -> 'XANES':
    """
    Reads supported XANES spectrum files (e.g., FDMNES .txt output) and returns
    a xanesnet.XANES object.

    Args:
        filepath (Union[str, Path]): XANES spectrum file.
        format(str, optional): File format for reading energy/absorption data
            (e.g., .csv, .txt, .bav, etc.). If `None`, the file format is
            determined from the file extension. Defaults to `None`.

    Returns:
        XANES: XANES spectrum.
    """
    
    filepath = Path(filepath)

    if format is None:
        format = filepath.suffix.lstrip('.')

    readers = {
        'csv': _read_from_csv,
        'txt': _read_from_txt,
        'bav': _read_from_bav
    }
    
    with open(filepath, 'r') as f:
        try:
            return readers[format](f)
        except KeyError:
            raise ValueError(
                f'{format} is not a supported file format for reading energy/'
                'absorption data'
            ) from None

def write(
    filepath: Union[str, Path],
    spectrum: 'XANES',
    format: str = 'csv'
) -> None:
    """
    Writes energy/absorption data from a xanesnet.XANES object in supported
    file formats (e.g., .csv, .txt, etc.).

    Args:
        filepath (Union[str, Path]): XANES spectrum file.
        spectrum (XANES): XANES spectrum.
        format (str, optional): File format for writing energy/absorption data
            (e.g., .csv, .txt, etc.). Defaults to 'csv'.

    Raises:
        ValueError: If the file format for writing energy/absorption data is
            not supported.
    """
    
    filepath = Path(filepath)

    writers = {
        'csv': _write_as_csv,
        'txt': _write_as_txt
    }

    with open(filepath, 'w') as f:
        try:
            writers[format](f, spectrum)
        except KeyError:
            raise ValueError(
                f'{format} is not a supported file format for writing energy/'
                'absorption data'
            ) from None

def _read_from_csv(
    f: TextIO
) -> 'XANES':
    """
    Reads energy/absorption data from a .csv file and returns a xanesnet.XANES
    object.

    Args:
        f (TextIO): .csv file to read energy/absorption data from.

    Returns:
        XANES: XANES spectrum.
    """
    
    energy, absorption = np.loadtxt(
        f, delimiter = ',', unpack = True
    )

    return XANES(energy, absorption)

def _read_from_txt(
    f: TextIO
) -> 'XANES':
    """
    Reads energy/absorption data from an FDMNES-style (.txt) file and returns a
    xanesnet.XANES object.

    Args:
        f (TextIO): FDMNES-style .txt file to read energy/absorption data from.

    Returns:
        XANES: XANES spectrum.
    """

    absorption_edge = float(f.readline().split()[0])
    energy, absorption = np.loadtxt(
        f, skiprows = 1, unpack = True
    )

    return XANES(energy, absorption, absorption_edge = absorption_edge)

def _read_from_bav(
    f: TextIO
) -> 'XANES':
    
    raise NotImplementedError(
        'support for reading energy/absorption data from FDMNES `_bav.txt` '
        'files is coming in a future version of XANESNET'
    )

def _write_as_csv(
    f: TextIO,
    spectrum: 'XANES'
) -> None:
    """
    Writes energy/absorption data from a xanesnet.XANES object in .csv format.

    Args:
        f (TextIO): XANES spectrum file to write energy/absorption data to.
        spectrum (XANES): XANES spectrum.
    """
    
    data = np.column_stack((spectrum.energy, spectrum.absorption))
    header = (
        'energy,absorption'
    )
    np.savetxt(
        f, data, delimiter = ',', header = header, fmt = ('%.3f', '%.6f')
    )

def _write_as_txt(
    f: TextIO,
    spectrum: 'XANES'
) -> None:
    """
    Writes energy/absorption data from a xanesnet.XANES object in 'mock'
    FDMNES-style (.txt) format.

    Args:
        f (TextIO): XANES spectrum file to write energy/absorption data to.
        spectrum (XANES): XANES spectrum.
    """
    
    data = np.column_stack((spectrum.energy, spectrum.absorption))
    header = (
        f'  {spectrum.absorption_edge:.3f} = e_edge\n  energy      <xanes>'
    )
    np.savetxt(
        f, data, delimiter = '  ', header = header, fmt = ('%10.2f', '%15.7e')
    )
