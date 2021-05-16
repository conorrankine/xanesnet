<h1 align = "center">
    <br>
        <img src = "./resources/xanesnet_graphic.png" width = "300">
    <br>
        <strong>
            X A N E S N E T
        </strong>
    <br>
</h1>

<p align = "center">
    <a href="https://linkedin.com/in/conorrankine">Dr. Conor D. Rankine </a>
    <br>
    <a href="https://ncl.ac.uk/nes/people/profile/tompenfold.html">Dr. Tom J. Penfold </a>
</p>

<p align = "center">
    <a href="http://penfoldgroup.co.uk">Penfold Group </a> @ <a href="https://ncl.ac.uk">Newcastle University </a>
</p>

<p align = "center">
    <a href="#setup">Setup</a> • <a href="#quickstart">Quickstart</a> • <a href="#publications">Publications</a>
</p>

#

We think that the theoretical simulation of X-ray spectroscopy (XS) should be fast, affordable, and accessible to all researchers. 

The popularity of XS is on a steep upward trajectory globally, driven by advances at, and widening access to, high-brilliance light sources such as synchrotrons and X-ray free-electron lasers (XFELs). However, the high resolution of modern X-ray spectra, coupled with ever-increasing data acquisition rates, brings into focus the challenge of accurately and cost-effectively analyzing these data. Decoding the dense information content of modern X-ray spectra demands detailed theoretical calculations that are capable of capturing satisfactorily the complexity of the underlying physics but that are - at the same time - fast, affordable, and accessible enough to appeal to researchers. 

This is a tall order - but we're using deep neural networks to make this a reality. 

XANESNET is under continuous development, so feel free to flag up any issues/make pull requests - we appreciate your input!

#

## SETUP

The quickest way to get started with XANESNET is to clone this repository:

```
git clone https://www.gitlab.com/conor.rankine/xanesnet
```

...change directory:

```
cd ./xanesnet
```

...and install XANESNET with pip:

```
pip install .
```

Now you're good to go!

#

## QUICKSTART

There are two core routines that you can launch *via* the command-line interface: ```learn``` and ```predict```.

### LEARN

The ```learn``` routine loads and preprocesses geometric (**X**; .xyz files with the absorption site as the first atom in the system) and X-ray spectroscopic (**Y**; .txt files formatted as output from the FDMNES program package) data, sets up a neural network, and fits the neural network to minimise the loss for mapping **Y** <- **X**. You can launch the ```learn``` routine with:

```
python3 -m xanesnet.cli learn <in.txt>
```

...where ```<in.txt>``` is the path to an input file containing definitions of all the necessary variables to launch the ```learn``` routine; variables are given on separate lines with the syntax: ```var = val```, and blank lines/lines starting with # are ignored/interpreted as comments. You can find example input files and data in this repository under ```examples```.

The ```learn``` routine produces a model.xxxx directory in the current workspace containing outputs, performance logs, and serialised objects for restoring the optimised neural network in the future.

You can see all the available options for the ```learn``` routine with:

```
python3 -m xanesnet.cli learn -h
```

### PREDICT

The ```predict``` routine uses a neural network that has previously been fit by the ```learn``` routine to predict the X-ray absorption near-edge structure (XANES) spectra of arbitary absorption site geometries. You can launch the ```predict``` routine with:

```
python3 -m xanesnet.cli predict <model_dir> <xyz_dir>
```

...where ```<model_dir>``` is the path to the model.xxxx output directory produced by the ```learn``` routine and ```<xyz_dir>``` is the path to a directory containing .xyz files (with the absorption site as the first atom in the system) to predict the XANES spectra for. 

The ```predict``` routine produces a predict.xxxx directory in the current workspace containing the predicted XANES spectra in .csv format.

It is also possible to convolute/broaden the predicted XANES spectra to account for core-hole lifetime broadening, instrumental resolution, and many-body effects (*e.g.* inelastic losses) by additionally passing the ```predict``` routine a convolution input file containing definitions of all the necessary variables, *e.g.*:

```
python3 -m xanesnet.cli predict <model_dir> <xyz_dir> -c <conv_in.txt>
```

...where ```<conv_in.txt>``` is the path to the convolution input file. It takes the same format as ```<in.txt>``` and, similarly, you can find examples in this repository under ```examples```. Convoluted XANES spectra will also be put into the predict.xxxx directory, and are distinguished by the '_conv' suffix.

You can see all the available options for the ```predict``` routine with:

```
python3 -m xanesnet.cli predict -h
```

#

## PUBLICATIONS

### The Program:
*[A Deep Neural Network for the Rapid Prediction of X-ray Absorption Spectra](https://doi.org/10.1021/acs.jpca.0c03723)* - C. D. Rankine, M. M. M. Madkhali, and T. J. Penfold, *J. Phys. Chem. A*, 2020, **124**, 4263-4270.

### The Applications:
*[Enhancing the Anaysis of Disorder in X-ray Absorption Spectra: Application of Deep Neural Networks to T-Jump-X-ray Probe Experiments](https://doi.org/10.1039/D0CP06244H)* - M. M. M. Madkhali, C. D. Rankine, and T. J. Penfold, *Phys. Chem. Chem. Phys.*, 2021, **23**, 9259-9269.

### Miscellaneous:
*[The Role of Structural Representation in the Performance of a Deep Neural Network for X-ray Spectroscopy](https://doi.org/10.3390/molecules25112715)* - M. M. M. Madkhali, C. D. Rankine, and T. J. Penfold, *Molecules*, 2020, **25**, 2715.

#