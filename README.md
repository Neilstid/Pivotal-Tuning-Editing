#   Pivotal Tuning Editing: Towards Disentangled Wrinkle Editing with GANs

____
*Paper:*

____

## Abstract

><p align=justify>Generative Adversarial Networks (GANs) enable image editing by manipulating image features. However, these manipulations still lack disentanglement. For example, when a specific wrinkle is edited, other age-related features or facial expressions are often changed as well. This paper proposes a new method for disentangled editing. The presented approach is based on two pivot images that allow learning an editing direction for an input image. These pivots are based on a real image (the input) and a synthetic modification of the real image along the desired editing direction. Although our primary focus is on wrinkle editing applications, our method can be extended to other editing tasks, such as hair color or lipstick editing. Qualitative and quantitative results show that our Pivotal Tuning Editing (PTE) provides a higher level of disentanglement and a more realistic editing than state-of-theart methods. </p>

![Alt text](./misc/exemple_editing.svg)


## Installation

![Static Badge](https://img.shields.io/badge/build-3.10.12-rgb(255%2C%20225%2C%2095)?style=flat&logo=python&logoColor=rgb(223%2C%20223%2C%20223)&label=python&labelColor=rgb(61%2C%20122%2C%20171)&link=https%3A%2F%2Fwww.python.org%2Fdownloads%2Frelease%2Fpython-31012%2F)
![Static Badge](https://img.shields.io/badge/build-11.7-rgb(116%2C%20183%2C%2027)?style=flat&logo=nvidia&logoColor=rgb(223%2C%20223%2C%20223)&label=cuda&labelColor=rgb(0%2C%200%2C%200)&link=https%3A%2F%2Fdeveloper.nvidia.com%2Fcuda-11-7-0-download-archive)

To install all package needed by using pip:

    pip install -r requirements.txt

Or by using conda:

    conda env create -f environment.yml 

## Usage

To train on your own images, go check the [Jupyter Notebook](./demo.ipynb). You can also use a pre-trained exemple by running the [app](./app.py).

## Contact

Feel free to contact me for any question on the repository or on the published article on [FG24](https://fg2024.ieee-biometrics.org/) using the issues section or my emails (neil99.farmer@gmail.com or neil.farmer@chanel.com).