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

import numpy as np
from pathlib import Path
from typing import Union, TextIO
from .templates import BaseTransformer

###############################################################################
################################## CLASSES ####################################
###############################################################################

class XANES():

    def __init__(
        self,
        e: np.ndarray,
        m: np.ndarray,
        e0: float = None,
        info: dict = None
    ):
        """
        Args:
            e (np.ndarray; 1D): an array of energy (`e`; eV) values
            m (np.ndarray; 1D): an array of intensity (`m`; arbitrary) values
            e0 (float, optional): the X-ray absorption edge (`e0`; eV) energy.
                If None, an attempt is made to determine `e0` from the maximum
                derivative of `m` with `get_e0()`.
                Defaults to None.
            info (dict, optional): a dictionary of key/val pairs that can be
                used to store extra information about the XANES spectrum as a
                tag.
                Defaults to None.

        Raises:
            ValueError: if the `e` and `m` arrays are not the same length.
        """
        
        if len(e) == len(m):
            self._e = e
            self._m = m
        else:
            raise ValueError('the energy (`e`) and XANES spectral intensity '\
                '(`m`) arrays are not the same length')

        if e0 is not None:
            self._e0 = e0
        else:
            self._e0 = self.estimate_e0()

        if info is not None:
            self.info = info
        else:
            self.info = {}

    def estimate_e0(self) -> float:
        """
        Estimates the energy of the X-ray absorption edge (`e0`; eV) as the
        energy at which the derivative of the absorption intensity is largest.

        Returns:
            float: Estimated energy of the X-ray absorption edge (in eV).
        """

        return self._e[np.argmax(np.gradient(self._m))]

    def scale(
        self,
        fit_limits: tuple = (100.0, 400.0),
        flatten: bool = True
    ):
        """
        Scales the XANES spectrum using the 'edge-step' approach, i.e., by
        fitting a 2nd-order (quadratic) polynomial to (part of) the post-edge
        (where `energy` >= `e0`), determining the 'edge step', `fit(e0)`, and
        scaling the absorption intensity by dividing through by this value.
         
        Optionally, the XANES spectrum can also be flattened post-scaling; the
        post-edge can be levelled off to ca. 1.0 by adding (1.0 - `fit(e0)`)
        to the absorption intensity where `energy` >= `e0`.

        Args:
            fit_limits (tuple, optional): Limits (lower and upper; in eV
                relative to the X-ray absorption edge) that bound the energy
                window over which the 2nd-order (quadratic) polynomial is fit.
                Defaults to (100.0, 400.0).
            flatten (bool, optional): Toggles flattening of the post-edge by
                adding (1.0 - `fit(e0)`) to the absorption intensity where
                `energy` >= `e0`. Defaults to True.
        """

        e_rel = self._e - self._e0
        e_rel_min, e_rel_max = fit_limits

        fit_window = (e_rel >= e_rel_min) & (e_rel <= e_rel_max)

        fit = np.polynomial.Polynomial.fit(
            self._e[fit_window],
            self._m[fit_window],
            deg = 2
        )

        self._m /= fit(self._e0)

        if flatten:
            self._m[self._e >= self._e0] += (
                1.0 - (fit(self._e)[self._e >= self._e0] / fit(self._e0))
            )

        return self

    def convolve(
        self,
        conv_type: str = 'fixed_width',
        conv_params: dict = None
    ):
        """
        Convolves the XANES spectrum with a fixed-width or energy-dependent
        (variable-width) Lorentzian function where the widths are
        derived from a convolution model (e.g. Seah-Dench; arctangent; etc.).

        XANES spectra are projected onto an auxilliary energy grid via linear
        interpolation pre-convolution; the spacing of the energy gridpoints
        on the auxilliary energy grid are equal to the smallest spacing of the
        energy gridpoints on the original energy grid.

        Args:
            conv_type (str, optional): Convolution type; options are
                'fixed_width', 'seah_dench', and 'arctangent'. Defaults to
                'fixed_width'.
            conv_params (dict, optional): Convolution parameters passed to
                the convolution width calculation function. If not set, `width`
                defaults to 2.0 eV and `ef` (the Fermi energy relative to the
                X-ray absorption edge) defaults to -1.0 eV. Defaults to None.

        Raises:
            ValueError: If `conv_type` is not one of 'fixed_width',
                'seah_dench', or 'arctangent'.
        """

        if conv_type not in ('fixed_width', 'seah_dench', 'arctangent'):
            raise ValueError(
                f'convolution type {conv_type} is not a recognised/supported '
                'convolution type; try, e.g., \'fixed_width\', '
                '\'seah_dench\', or \'arctangent\''                
            )

        if conv_params is None:
            conv_params = {}

        conv_params.setdefault('width', 2.0)
        conv_params.setdefault('ef', -1.0)

        de = np.min(np.diff(self._e))

        pad = de * int((50.0 * conv_params['width']) / de)

        e_aux = np.linspace(
            np.min(self._e) - pad,
            np.max(self._e) + pad,
            int((np.ptp(self._e) + (2.0 * pad)) / de) + 1
        )

        width = _get_conv_width(
            e_aux - self._e0,
            conv_type = conv_type,
            conv_params = conv_params
        )

        # remove cross-sectional contributions to `m` below `ef`
        self._m[self._e < (self._e0 + conv_params['ef'])] = 0.0

        # project `m` onto the auxilliary energy scale `e_aux`
        m_aux = np.interp(e_aux, self._e, self._m)

        # create the convolutional filter `conv_filter`
        conv_filter = _get_conv_filter(e_aux, width)

        # convolve `m_aux` with the convolution filter `conv_filter`
        m_aux = np.sum(conv_filter * m_aux, axis = 1)

        # project `m_aux` onto the original energy scale `e`
        self._m = np.interp(self._e, e_aux, m_aux)

        return self

    @property
    def e(self) -> float:
        return self._e

    @property
    def m(self) -> float:
        return self._m

    @property
    def e0(self) -> float:
        return self._e0

    @property
    def spectrum(self) -> tuple:
        return (self._e, self._m)

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
        e_min: float = -30.0,
        e_max: float = 120.0,
        n_bins: int = 150,
        scale: bool = True,
        conv_type: str = None,
        conv_params: dict = None
    ):
        """
        Args:
            e_min (float, optional): Minimum energy bound (in eV) for the
                spectral window/slice relative to the X-ray absorption edge.
                Defaults to -30.0.
            e_max (float, optional): Maximum energy bound (in eV) for the
                spectral window/slice relative to the X-ray absorption edge.
                Defaults to 120.0.
            n_bins (int, optional): Number of discrete energy bins for the
                spectral window/slice. Defaults to 150.
            scale (bool, optional): Toggles spectrum scaling using the
                'edge-step' approach. Defaults to `True`.
            conv_type (str, optional): Convolution type; options are
                'fixed_width', 'seah_dench', and 'arctangent'. If None,
                spectra are not convolved. Defaults to None.
            conv_params (dict, optional): Convolution parameters passed through
                to the convolution width calculation function. Defaults to
                None.
        """
        
        # TODO: sanity-check inputs and raise errors if necessary
        self._e_min = e_min
        self._e_max = e_max
        self._n_bins = n_bins
        self._e_aux = np.linspace(
            self._e_min, self._e_max, self._n_bins
        )

        self.scale = scale

        self.conv_type = conv_type
        self.conv_params = conv_params

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

        if self.conv_type:
            spectrum.convolve(
                conv_type = self.conv_type,
                conv_params = self.conv_params
            )

        if self.scale:
            spectrum.scale()

        spectrum_ = np.interp(
            self._e_aux, spectrum.e - spectrum.e0, spectrum.m
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
    e_rel: np.ndarray,
    width: float = 2.0,
    width_max: float = 15.0,
    ef: float = -5.0,
    a: float = 1.0
) -> np.ndarray:
    """
    Calculates the widths of the energy-dependent Lorentzians under the Seah-
    Dench convolution model.

    Args:
        e_rel (np.ndarray): Energies (in eV) relative to the X-ray absorption
            edge.
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

    e_ = (e_rel - ef)

    return width + (
        (a * width_max * e_) / (width_max + (a * e_))
    )

def _calc_arctangent_conv_width(
    e_rel: np.ndarray,
    width: float = 2.0,
    width_max: float = 15.0,
    ef: float = -5.0,
    ec: float = 30.0,
    el: float = 30.0
) -> np.ndarray:
    """
    Calculates the widths of the energy-dependent Lorentzians under the
    arctangent convolution model.

    Args:
        e_rel (np.ndarray): Energies (in eV) relative to the X-ray absorption
            edge.
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
   
    e_ = (e_rel - ef) / ec
    with np.errstate(divide = 'ignore'):
        arctan = (np.pi / 3.0) * (width_max / el) * (e_ - (1.0 / e_**2))

    return width + (
        width_max * ((1.0 / 2.0) + (1.0 / np.pi) * np.arctan(arctan))
    )

def _get_conv_width(
    e_rel: np.ndarray,
    conv_type: str,
    conv_params: dict
) -> Union[float, np.ndarray]:
    """
    Returns the width(s) of the fixed-width or energy-dependent Lorentzian(s)
    under the specified type of convolutional model.

    Args:
        e_rel (np.ndarray): Energies (in eV) relative to the X-ray absorption
            edge.
        conv_type (str): Convolution type; options are 'fixed_width',
            'seah_dench', and 'arctangent'.
        conv_params (dict): Convolution parameters passed through to the
            convolution width calculation function.

    Raises:
        ValueError: If `conv_type` is not one of 'fixed_width', 'seah_dench',
            or 'arctangent'.

    Returns:
        Union[float, np.ndarray]: Width of the Lorentzian if `conv_type` is
            'fixed width', else an array of values corresponding to the widths
            of the energy-dependent Lorentzians under the specified type of
            convolutional model if `conv_type` is 'seah_dench' or 'arctangent'.
    """
    
    if conv_params is None:
        conv_params = {}

    if conv_type == 'fixed_width':
            return conv_params['width']
    elif conv_type == 'seah_dench':
        return _calc_seah_dench_conv_width(
            e_rel = e_rel, **conv_params
        )
    elif conv_type == 'arctangent':
        return _calc_arctangent_conv_width(
            e_rel = e_rel, **conv_params
        )
    else:
        raise ValueError(
            f'convolution type {conv_type} is not a recognised/supported '
            'convolution type; try, e.g., \'fixed_width\', '
            '\'seah_dench\', or \'arctangent\''
        )

def _get_conv_filter(
    e_rel: np.ndarray,
    width: Union[float, np.ndarray]
) -> np.ndarray:
    """
    Returns a (2D) array of Lorentzian convolutional filters with variable
    centers and (optionally) widths defined over a (1D) energy grid; each
    Lorentzian convolutional filter is centered at a different energy
    gridpoint, i.e., the i^{th} row of the array of Lorentzian convolutional
    filters contains a Lorentzian function defined over `x` = `e_rel` with
    the center `x0` = `e_rel[i]` and `width` = `width[i]` (else `width` =
    `width` if `width` is a float rather than a np.ndarray).

    Args:
        e_rel (np.ndarray): Energies (in eV) relative to the X-ray absorption
            edge.
        width (Union[float, np.ndarray]): Width(s) of the Lorentzians (in
            eV); if scalar (float), all Lorentzians will have constant width,
            else, if a np.ndarray, Lorentzians will have variable width (e.g.,
            the i^{th} Lorentzian will have a width = `width[i]`).

    Returns:
        np.ndarray: (2D) Array of Lorentzian convolutional filters. 
    """
    
    return _lorentzian(
        np.meshgrid(e_rel, e_rel), width
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

    e0 = float(f.readline().split()[0])
    energy, absorption = np.loadtxt(
        f, skiprows = 1, unpack = True
    )

    return XANES(energy, absorption, e0 = e0)

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
    
    data = np.column_stack((spectrum.e, spectrum.m))
    header = 'energy,absorption'
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
    
    data = np.column_stack((spectrum.e, spectrum.m))
    header = f'  {spectrum.e0:.3f} = e_edge\n  energy      <xanes>'
    np.savetxt(
        f, data, delimiter = '  ', header = header, fmt = ('%10.2f', '%15.7e')
    )
