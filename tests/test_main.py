from cgi import test
from breeding_gym import main
import pytest

def test_main():
    main(OUTFILE = "Outfilex.txt", BS = 10, NOF = 50, SPLIT= 0.2)


if __name__ == "__main__":
    test_main()