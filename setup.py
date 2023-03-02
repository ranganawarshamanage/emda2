from __future__ import division, absolute_import, print_function
import setuptools

try:
    from numpy.distutils.core import setup, Extension
except ImportError:
    print("Numpy is not installed! "
          "Please install Numpy first and then try")
    raise SystemExit()

ex1 = Extension(name = 'fcodes2', 
                sources = ['emda2/fcodes2.f90'])

version = {}
with open("./emda2/config.py") as fp:
    exec(fp.read(), version)

setup(name='emda2',
    version=version['__version__'],
    description= 'Electron Microscopy map and model manipulation tools',
    #url='https://www2.mrc-lmb.cam.ac.uk/groups/murshudov/content/emda/emda.html',
    author='Rangana Warshamanage, Garib N. Murshudov',
    author_email='ranganaw@mrc-lmb.cam.ac.uk, garib@mrc-lmb.cam.ac.uk',
    license='MPL-2.0',
    packages=setuptools.find_packages(),
    #ext_modules =[ex1],
    install_requires=[
        'pandas',
        'mrcfile',
        'matplotlib',
        'scipy',
        'gemmi',
        #'servalcat',
        'tabulate',
        'more_itertools',
        #'proshade',
        'proshade @ git+https://github.com/GaribMurshudov/proshade_mod1.git'
        'scikit-image'
        ],
    ext_modules =[ex1],
    #test_suite='emda.tests',
    entry_points={
      'console_scripts': [
          #'emda_test = emda.emda_test:main',
          #'emda_test_exhaust = emda.emda_test_exhaust:main',
          'emda2 = emda2.emda_cli:main',
                          ],
      },
    zip_safe= False)
