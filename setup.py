from setuptools import setup

setup(
    name='gallop',
    version='0.1.3',
    description='Fast crystal structure determination from powder diffraction data',
    url='https://github.com/mspillman/gallop',
    author='Mark Spillman',
    author_email='markspillman@gmail.com',
    license='GPLv3',
    packages=['gallop', 'gallop.optim', 'gallop.files'],
    package_data={'gallop': ['gallop/example_data/*.*',
                            'gallop/user_settings/Default.json',
                            'gallop/atomic_scattering_params.json']},
    include_package_data=True,
    scripts=['gallop_streamlit.py', 'gallop.bat'],
    install_requires=['pymatgen==2023.5.10',
                        'numpy==1.24.3',
                        'torch==2.0.0',
                        'scipy==1.10.1',
                        'matplotlib',
                        'torch-optimizer==0.3.0',
                        'streamlit==1.11.1',
                        'pandas==1.5.3',
                        'altair==4',
                        'py3Dmol==2.0.1.post1',
                        'mpld3==0.5.9',
                        ],

)
