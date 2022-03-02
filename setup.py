from setuptools import setup
import punctatools

setup(
    name='punctatools',
    version=punctatools.__version__,
    author="Anna Medyukhina",
    author_email='anna.medyukhina@gmail.com',
    packages=['punctatools',
              'punctatools.lib'
              ],
    license='Apache License 2.0',
    include_package_data=True,

    test_suite='punctatools.tests',

    install_requires=[
        'setuptools>=18.5,<=57.5.0',
        'cellpose',
        'ipykernel',
        'scipy',
        'ddt',
        'pytest',
        'tqdm',
        'scikit-image',
        'pandas',
        'seaborn',
        'bokeh<2.5.0,>=2.4.0',
        'holoviews',
        'jupyter',
        'am_utils @ git+https://github.com/amedyukhina/am_utils.git',
        'intake_io @ git+https://github.com/bhoeckendorf/intake_io.git@v0.0.2',
    ],
)
