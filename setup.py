import setuptools

install_requires = [
    'numpy',
    'pandas',
    'matplotlib',
    'gym>=0.26',
    'ipykernel',
]


setuptools.setup(
    name='breedinggym',
    version='0.1.0',
    description='Open AI gym interface to the cycles crop simulator',
    url='https://gitlab.inf.ethz.ch/lucac/breedinggym',
    author='Omar Younis, Luca Corinzia, Matteo Turchetta',
    author_email='luca.corinzia@inf.ethz.ch',
    keywords='Crop breeding simulator',
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    include_package_data=True,
    install_requires=install_requires,
    extras_require={},
    cmdclass={},
)
