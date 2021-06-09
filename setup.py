import subprocess
import sys

from setuptools import setup
# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext

__version__ = "0.0.1"

PKG_NAME: str = 'fast_gsdmm'

# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.

ext_modules = [
    Pybind11Extension(PKG_NAME,
                      ["src/main.cpp"],
                      define_macros=[('VERSION_INFO', __version__)],
                      cxx_std=17
                      ),
]

with open('README.md', 'r') as _read_me:
    long_description = _read_me.read()

setup(
    name=PKG_NAME,
    version=__version__,
    author="Gianni Francesco Balistreri",
    author_email="gbalistreri@gmx.de",
    description="Gibbs Sampling Dirichlet Multinomial Modeling algorithm for short-text clustering",
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='GNU',
    url='https://github.com/GianniBalistreri/gsdmm',
    ext_modules=ext_modules,
    #extras_require={"test": "pytest"},
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    #cmdclass={"build_ext": build_ext},
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)

subprocess.run(['python{} -m pip install {}'.format('3' if sys.platform.find('win') != 0 else '',
                                                    PKG_NAME
                                                    )
                ], shell=True)
