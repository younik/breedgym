import setuptools

install_requires = [
    'numpy',
    'pandas',
    'matplotlib',
    'gymnasium>=0.29',
    'jax',
    'jaxtyping'
]


setuptools.setup(
    name='breedgym',
    version='0.0.1a',
    description='Gym environment for breeding simulation',
    url='https://gitlab.inf.ethz.ch/lucac/breedgym',
    author='Omar Younis, Luca Corinzia, Matteo Turchetta',
    author_email='omargallalaly.younis@inf.ethz.ch',
    keywords='Plant breeding simulator for Reinforcement Learning',
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    include_package_data=True,
    install_requires=install_requires,
    extras_require={},
    cmdclass={},
)
