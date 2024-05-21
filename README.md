#   Pivotal Tuning Editing: Towards Disentangled Wrinkle Editing with GANs


![Alt text](./misc/exemple_editing.svg)



## Abstract

Generative Adversarial Networks (GANs) enable image editing by manipulating image features. However, these manipulations still lack disentanglement. For example, when a specific wrinkle is edited, other age-related features or facial expressions are often changed as well. This paper proposes a new method for disentangled editing. The presented approach is based on two pivot images that allow learning an editing direction for an input image. These pivots are based on a real image (the input) and a synthetic modification of the real image along the desired editing direction. Although our primary focus is on wrinkle editing applications, our method can be extended to other editing tasks, such as hair color or lipstick editing. Qualitative and quantitative results show that our Pivotal Tuning Editing (PTE) provides a higher level of disentanglement and a more realistic editing than state-of-theart methods.

## Installation

Tested version of python: 3.10.12 \
Tested version of CUDA: 11.7

To install all package needed:

    pip install -r requirements.txt

## Usage

To train on your own images, go check the [Jupyter Notebook](./demo.ipynb). You can also use a pre-trained exemple by running the [app](./app.py).

## Contact

Feel free to contact me for any question on the repository or on the published article on [FG24](https://fg2024.ieee-biometrics.org/) using the issues section or my emails (neil99.farmer@gmail.com or neil.farmer@chanel.com).