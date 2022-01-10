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
        'scipy',
        'ddt',
        'pytest',
        'tqdm',
        'scikit-image',
        'pandas',
        'seaborn',
        'am_utils',
        'bokeh<2.5.0,>=2.4.0',
        'holoviews',
        'jupyter'
    ],
    dependency_links=[
        "https://github.com/amedyukhina/am_utils/releases/",
    ],
)
