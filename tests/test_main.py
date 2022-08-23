from breeding_gym import main
import os
import numpy as np

def test_main_1():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    main(OUTFILE = f"{dir_path}/Outfile_s7_10_50_10.txt", BS = 10, NOF = 50, SPLIT=1.0, seed=7)

    outfile = np.loadtxt(f"{dir_path}/Outfile_s7_10_50_10.txt")
    expected_outfile = np.loadtxt(f"{dir_path}/expected_output/Outfile_s7_10_50_10.txt")
    assert np.all(outfile == expected_outfile)

    offspring = np.loadtxt(f"{dir_path}/Offspring_s7_10_50_10.txt", dtype='O', delimiter=',')
    expected_offspring = np.loadtxt(f"{dir_path}/expected_output/Offspring_s7_10_50_10.txt", dtype='O', delimiter=',')
    assert np.all(offspring == expected_offspring)

    os.remove(f"{dir_path}/Outfile_s7_10_50_10.txt")
    os.remove(f"{dir_path}/Offspring_s7_10_50_10.txt")


def test_main_0():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    main(OUTFILE = f"{dir_path}/Outfile_s7_10_50_0.txt", BS = 10, NOF = 50, SPLIT=0, seed=7)

    outfile = np.loadtxt(f"{dir_path}/Outfile_s7_10_50_0.txt")
    expected_outfile = np.loadtxt(f"{dir_path}/expected_output/Outfile_s7_10_50_0.txt")
    assert np.all(outfile == expected_outfile)

    offspring = np.loadtxt(f"{dir_path}/Offspring_s7_10_50_0.txt", dtype='O', delimiter=',')
    expected_offspring = np.loadtxt(f"{dir_path}/expected_output/Offspring_s7_10_50_0.txt", dtype='O', delimiter=',')
    assert np.all(offspring == expected_offspring)

    os.remove(f"{dir_path}/Outfile_s7_10_50_0.txt")
    os.remove(f"{dir_path}/Offspring_s7_10_50_0.txt")

def test_main_02():
    # TODO fix, this throw an error aso in original
    pass