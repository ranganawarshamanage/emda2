from __future__ import division, absolute_import, print_function
import setuptools
from numpy.distutils.core import setup, Extension

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
    install_requires=['pandas>=0.23.4','mrcfile','matplotlib','numpy','scipy','gemmi','servalcat'],
    ext_modules =[ex1],
    #test_suite='emda.tests',
    zip_safe= False)
