from setuptools import setup, find_packages

setup(
    name='gallop',
    version='0.1.0',
    description='Fast crystal structure determination from powder diffraction data',
    url='https://github.com/mspillman/gallop',
    author='Mark Spillman',
    author_email='markspillman@gmail.com',
    license='GPLv3',
    packages=[find_packages()],
    install_requires=['pymatgen==2021.2.8.1',
                        'numpy',
                        'torch',
                        'scipy',
                        'matplotlib',
                        'torch_optimizer',
                        'streamlit',
                        'pandas',
                        'altair',
                        'py3Dmol',
                        ],

)
