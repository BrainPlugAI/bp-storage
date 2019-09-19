'''
Setup Script for the Storage Module in the labeling pipe.

This will install the charon module to the local python distribution.
Licence Information:
https://creativecommons.org/licenses/by-nc-nd/3.0/
'''


from setuptools import setup
from setuptools import find_packages


__status__      = "Package"
__copyright__   = "Copyright 2019, BrainPlug"
__license__     = "MIT License"
__version__     = "1.0.1"

# 01101100 00110000 00110000 01110000
__author__      = "Felix Geilert"
__email__       = "f.geilert@brainplug.de"


def readme():
    '''Retrieves the content of the readme.'''
    with open('readme.md') as f:
        return f.read()


setup(name='bp_storage',
      version=__version__,
      description='Helper Library for various Dataset formats',
      long_description=readme(),
      long_description_content_type="text/markdown",
      classifiers=['Programming Language :: Python :: 3.5',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence'],
      keywords='storage datasets deep learning',
      url='https://github.com/BrainPlugAI/bp-storage',
      author='Felix Geilert',
      author_email='f.geilert@brainplug.de',
      license='MIT License',
      packages=find_packages(),
      install_requires=[ 'numpy', 'imgaug' ],
      include_package_data=True,
      zip_safe=False)
