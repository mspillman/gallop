# GALLOP
Gradient Accelerated LocaL Optimsation and Particle Swarm: a fast method for crystal structure determination from powder diffraction data.

## Installation

### Requirements
Installation instructions here

------------------------------

## Running GALLOP
### WebApp
Instructions to run webapp here

### Python mode
Instructions to run in Python mode here

------------------------------

## References, useful libraries and links
If you make use of GALLOP in your work, please cite the following papers:
- Spillman and Shankland 1
- Spillman and Shankland 2

### Papers
[Internal to Cartesian](https://pubmed.ncbi.nlm.nih.gov/15898109/) - GALLOP uses the Natural Extension Reference Frame method for converting internal to Cartesian coordinates.

[Correlated Integrated Intensity Chisqd](https://scripts.iucr.org/cgi-bin/paper?ks5013) - This is faster to calculate than Rwp and other goodness of fit metrics, but requires the inverse covariance matrix from a Pawley refinement so the Le Bail method of intensity extraction is not suitable.

[DASH](https://scripts.iucr.org/cgi-bin/paper?ks5103)

[GSAS-II](https://scripts.iucr.org/cgi-bin/paper?aj5212)

[TOPAS](https://scripts.iucr.org/cgi-bin/paper?jo5037)
### Libraries
GALLOP makes use of a number of libraries, without which its development would have been significantly more challenging. In particular:

[PyMatGen](https://pymatgen.org/) - the PyMatGen PXRD pattern calculator was used as the inspiration for what eventually became GALLOP. GALLOP still makes use of a number of PyMatGen features to handle things like space group symmetries.

[PyTorch](https://pytorch.org/) - the original code that eventually became GALLOP was originally written using numpy. PyTorch served as a near drop-in replacement that allowed automatic differentiation and running on GPUs/TPUs.

[Streamlit](https://streamlit.io/) - this allowed the WebApp to be written entirely in python, which made it significantly easier to integrate with the existing code.

### Other
[Google Colaboratory](https://colab.research.google.com) has been invaluable for the free/cheap access to GPUs and TPUs


