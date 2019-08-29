'''
Setup Script for the Storage Module in the labeling pipe.

This will install the charon module to the local python distribution.
Licence Information:
https://creativecommons.org/licenses/by-nc-nd/3.0/
'''


from setuptools import setup
from setuptools import find_packages


__status__      = "Package"
__copyright__   = "Copyright 2018, BrainPlug"
__license__     = "CC BY-NC-ND 3.0"
__version__     = "0.1.0"

# 01101100 00110000 00110000 01110000
__author__      = "Felix Geilert"
__email__       = "f.geilert@brainplug.de"


def readme():
    '''Retrieves the content of the readme.'''
    with open('readme.md') as f:
        return f.read()


setup(name='storage',
      version=__version__,
      description='Helper Library for various Dataset formats',
      long_description=readme(),
      classifiers=['Development Status :: 3 - Alpha',
                   'License :: Other/Proprietary License',
                   'Programming Language :: Python :: 3.5',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence'],
      keywords='storage datasets deep learning',
      url='https://bitbucket.org/cerebro_dev/bp-labeling',
      author='Felix Geilert',
      author_email='f.geilert@brainplug.de',
      license='CC BY-NC-ND 3.0',
      packages=find_packages(),
      install_requires=[ 'numpy', 'imgaug' ],
      #entry_points={'console_scripts': [
    #      'charon-pack=charon.scripts.pack:main',
    #      'charon-visualize=charon.scripts.visualize:main',
    #      'charon-stats=charon.scripts.stats:main'
      #]},
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      zip_safe=False)
