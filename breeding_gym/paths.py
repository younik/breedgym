import pathlib

__all__ = ['PROJECT_PATH', 'DATA_PATH', 'CODE_PATH']

PROJECT_PATH = pathlib.Path(__file__).parents[1]
CODE_PATH = pathlib.Path(__file__).parents[0]
DATA_PATH = CODE_PATH.joinpath('data')