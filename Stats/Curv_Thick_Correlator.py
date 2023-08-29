import nibabel.freesurfer.io as fsio
import numpy as np
import pandas as pd
import csv
import sys
from tqdm import tqdm

from Brain_Analyser import *
from Dataset_Loader import *

class Curv_Thick_Correlator:
    #combines different analysis for a set
    ### annots, overall mean, lh mean, rh mean
    
    def __init__(self, subj_file : str, thick_path : str, thick_path : str, curv_path :str):
        text_file = open("", "r")
        lines = text_file.readlines(subj_file)
        self.test_subs = [line[:-1] for line in lines]
        self.thick_path = thick_path
        self.curv_path = curv_path
        self.fs_thick_path = thick_path
    def generate_stats(self, maximal_nr_of_elems = -1):
        print("Generating Cooeficients")
        arr_right_correlation = []
        arr_left_correlation = []
        for subject_id in tqdm(self.test_subs):
            # Condition to break the loop
            if maximal_nr_of_elems != -1:
                maximal_nr_of_elems = maximal_nr_of_elems - 1
                if maximal_nr_of_elems == -1:
                    break
            left_thick_d = abs(np.load(self.thick_path + "/lh/"+ subject_id +".npy") - np.load(self.fs_thick_path + "/lh/"+ subject_id +".npy"))
            right_thick_d = abs(np.load(self.thick_path + "/rh/"+ subject_id +".npy") - np.load(self.fs_thick_path + "/rh/"+ subject_id +".npy"))

            left_curv = np.load(self.curv_path + "/lh/"+ subject_id +".npy")
            right_curv = np.load(self.curv_path + "/rh/"+ subject_id +".npy")

            arr_right_correlation.append(np.corrcoef(left_thick_d, left_curv)[(1,0)])
            arr_left_correlation.append(np.corrcoef(right_thick_d, right_curv)[(1,0)])
        print(np.mean(arr_right_correlation))
        print(np.mean(arr_left_correlation))

        print("Max")
        print(np.max(arr_right_correlation))
        print(np.max(arr_left_correlation))

        return
