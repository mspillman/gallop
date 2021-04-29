# GALLOP
**Gradient Accelerated LocaL Optimisation and Particle Swarm: a fast method for crystal structure determination from powder diffraction data.**

Contents:
- [Local Installation](#local-installation)
- [Using GALLOP](#using-gallop)
- [References and Resources](#references-and-resources)

------------------------------
## Local Installation
GALLOP is able to easily make use of cloud-based GPU resources, and as such, does not require a GPU to be available on a users machine to be used. However, some users may wish to install GALLOP locally. Whilst these instructions have only been tested on Windows, the libraries used are cross-platform and therefore it should be possible to install GALLOP on Linux or Mac OS environments. The below instructions assume a Windows-based system. The only major difference with other platforms will be the C++ build tools.

For optimal performance, an NVidia GPU is recommended. It may be possible to use some AMD GPUs, provided that [ROCm](https://pytorch.org/blog/pytorch-for-amd-rocm-platform-now-available-as-python-package/) is compatible with the GPU, though this has not been tested.

Administrator privileges may be required.

<br />

**GALLOP requires:**
<br />

| Dependency | Version |Comments|
|------------|---------|--------|
| [Python](https://www.anaconda.com/products/individual) | 3.8 | Other versions of Python 3 will probably also work. Anaconda distribution strongly recommended.|
| [CUDA](https://developer.nvidia.com/cuda-toolkit-archive)   | 10.2 or 11.x | CUDA 11 recommended for Ampere GPUs, though this has not yet been tested thoroughly |
| [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive)  | Compatible with CUDA 10.2 or 11.x | Login required. Not strictly necessary for GALLOP, but will allow pytorch to be used with more flexibility |
| [Visual C++ build tools](http://landinghub.visualstudio.com/visual-cpp-build-tools) | 14.0 | Needed for installation of some of the Python libraries. Linux or Mac users should install appropriate C++ build tools if prompted to do so during library installation (see below).|

<br />

Once the above are installed, several Python Libraries must also be installed. The use of [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or [venv](https://docs.python.org/3/library/venv.html) virtual environments is recommended but not required.

<br />

| Library |Comments|
|---------|--------|
| [PyTorch](https://pytorch.org/get-started/locally/) | Must be compatible with the version of CUDA installed. Installation via conda package manager recommended. |
| [PyMatGen](https://pymatgen.org/) | Needed for various crystallographic symmetry related functions. v2021.2.8.1 needed |
| [Torch Optimizer](https://github.com/jettify/pytorch-optimizer) | Allows for non-standard local optimisers such as DiffGrad |
| [pyDOE](https://pythonhosted.org/pyDOE/) | Latin-hypercube sampling for initial points |
| [Streamlit](https://streamlit.io/) | Needed for the WebApp |
| [tqdm](https://pypi.org/project/tqdm/) | Lightweight progress bars for non-WebApp use |

<br />

PyTorch should be installed first using the instructions on the PyTorch website. Once installed, test that it is recognising the local GPU by opening a Python prompt and running the following commands:
```python
>>> import torch
>>> print(torch.cuda.is_available())
```
If the command prints ```True``` then PyTorch has been successfully installed and is able to use the local GPU. If it prints False, then PyTorch is not able to find the locally installed GPU and installation should be tried again. Note that GALLOP will work using CPU-only PyTorch, but it will likely be extremely slow.

Once PyTorch is properly installed, the remaining libraries can be installed using the following command, run from powershell or command prompt.
```
pip install pymatgen==2021.2.8.1 torch_optimizer pyDOE streamlit tqdm
```
------------------------------

<br />

## Using GALLOP

### **PXRD Data preparation**
Helper functions are available within GALLOP to read Pawley fitting outputs from DASH, GSAS-II and TOPAS.

- DASH: follow the Pawley fitting procedure as normal, and ensure that the resultant ```.sdi, .dsl, .hcv``` and ```.tic``` files are available.

- GSAS-II: Pawley fit the data as normal. Once satisfied with the fit, unflag **all** parameters apart from the intensities (i.e. peak shape, unit cell, background etc). Reset the intensity values, then ensure that only the intensities will refine. Ensure that for this final refinement, the optimisation algorithm is set to *analytic Jacobian*. This is critical, as the default *Hessian* optimiser modifies the covariance matrix in ways that produce errors in GALLOP. After saving, GALLOP will read in the ```.gpx``` file.

- TOPAS: Pawley fit the data as normal. Once satisfied with the fit, unflag **all** refineable parameters in the ```.inp```, and delete the intensities (if present). Add the key word ```do_errors``` before the ```hkl_Is``` term, and add the key word ```C_matrix``` to the end of the ```.inp```. GALLOP will read in the resultant ```.out``` file.

### **Z-matrices**
GALLOP is able to read Z-matrices that have been produced by the ```MakeZmatrix.exe``` program that is bundled with DASH.

One commonly encountered error is when the resultant Z-matrix has refineable torsion angles defined in terms of one or more hydrogen atoms. To fix this issue, follow the following steps:
1. Produce a CIF of the structure from which the Z-matrix is being generated
2. Reorder the atoms in the CIF such that all hydrogen atoms are recorded after all non-hydrogen atoms.
3. Regenerate the Z-matrices with DASH/MakeZmatrix.exe

Other programs can in principle be used to produce Z-matrices suitable for GALLOP. For more information, see the gallop.z_matrix module documentation.

### **Run GALLOP via the Web App**
#### **Local operation:**
In the folder containing GALLOP code, open a command prompt and run the following command:
```
streamlit run .\gallop_streamlit.py
```
This will automatically open a browser window displaying the GALLOP Web App.

#### **Cloud operation:**
Use this [Colab Notebook to try the GALLOP Web App for free]().
You will need a Google account to run the notebook, and an ngrok authentication key, which can be obtained for free at https://ngrok.com/
Save a copy of the notebook to your own Google drive for easy access in the future.


### **Run GALLOP via Python scripts / Jupyter notebooks**
#### **Local operation:**

#### **Cloud operation:**
Use this [Colab Notebook to try GALLOP in Python mode for free]().
You will need a Google account to run the notebook.
Save a copy of the notebook to your own Google drive for easy access in the future.




------------------------------

## References and resources
If you make use of GALLOP in your work, please cite the following papers:
- Spillman and Shankland 1
- Spillman and Shankland 2

### **Relevant articles**
[Internal to Cartesian](https://pubmed.ncbi.nlm.nih.gov/15898109/) - GALLOP uses the Natural Extension Reference Frame method for converting internal to Cartesian coordinates.

[Correlated Integrated Intensity Chisqd](https://scripts.iucr.org/cgi-bin/paper?ks5013) - This is faster to calculate than R<sub>wp</sub> and other goodness of fit metrics, but requires the inverse covariance matrix from a Pawley refinement so the Le Bail method of intensity extraction is not suitable.

[DASH](https://scripts.iucr.org/cgi-bin/paper?ks5103)

[GSAS-II](https://scripts.iucr.org/cgi-bin/paper?aj5212)

[TOPAS](https://scripts.iucr.org/cgi-bin/paper?jo5037)
### **Python Libraries**
GALLOP makes use of a number of libraries, without which its development would have been significantly more challenging. In particular:

[PyMatGen](https://pymatgen.org/) - the PyMatGen PXRD pattern calculator was used as the inspiration for what eventually became GALLOP. GALLOP still makes use of a number of PyMatGen features to handle things like space group symmetries.

[PyTorch](https://pytorch.org/) - the original code that eventually became GALLOP was originally written using numpy. PyTorch served as a near drop-in replacement that allowed automatic differentiation and running on GPUs/TPUs.

[Streamlit](https://streamlit.io/) - this allowed the WebApp to be written entirely in python, which made it significantly easier to integrate with the existing code.

### **GPU/TPU Resources**
[Google Colaboratory](https://colab.research.google.com) has been invaluable for the free/cheap access to GPUs and TPUs. Colab Pro is ~$10 per month which give priority access to more powerful GPU resources amongst other benefits. A number of other services also allow free or cheap GPU access, for example:

- [Kaggle](https://www.kaggle.com/) - free access to GPUs and TPUs, run time capped at 30 hours / week

- [Paperspace](https://www.paperspace.com/) - free access to some GPUs, plus various paid options for more powerful GPUs

- [Vast.ai](https://vast.ai/) - cheap access to a wide variety of GPUs not offered by other services.

A wide variety of bigger commercial providers are also available.


