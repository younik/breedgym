import pathlib

__all__ = ['PROJECT_PATH', 'DATA_PATH', 'CODE_PATH']

PROJECT_PATH = pathlib.Path(__file__).parents[2]
CODE_PATH = pathlib.Path(__file__).parents[1]
SIMULATOR_PATH = CODE_PATH.joinpath('simulator')
DATA_PATH = SIMULATOR_PATH.joinpath('data')
FIGURE_PATH = PROJECT_PATH.joinpath('figures')
