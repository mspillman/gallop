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
)
