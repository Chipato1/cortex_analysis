import nibabel.freesurfer.io as fsio
import numpy as np
import pandas as pd
import csv
import sys
from tqdm import tqdm

from Stats.Brain_Analyser import *
from Stats.Dataset_Loader import *

class Long_Dat_Analyser:
    #combines different analysis for a set
    ### annots, overall mean, lh mean, rh mean
    
    def __init__(self,
                 thick_csv_path : str, 
                 long_id_file_path : str):
        self.long_id_file_path = long_id_file_path
        self.thick_csv_path = thick_csv_path


    # <Image_ID>.long.<PTID>
    def generate_deltas(self, id_path):
        df_ids = pd.read_csv(self.long_id_file_path)
        df_thick = pd.read_csv(self.thick_csv_pat)
        patient_ids = set(df_ids["PTID"].to_numpy())

        df_data = []
        for pat_id in patient_ids:
            deltas = []
            last_sample_key = None
            i = 0

            for index, row in df_ids.loc[df['PTID'] == pat_id].sort_values(by=['Month']).iterrows():
                img_id = row["IMAGEUID"]
                sample_key = str(img_id) + ".long." + pat_id
                if last_sample_key != None:
                    new_vals = df_thick.loc[df_thick['id'] == sample_key].to_numpy().squeeze()[1:]
                    old_vals = df_thick.loc[df_thick['id'] == last_sample_key].to_numpy().squeeze()[1:]

                    if len(new_vals) != 0 and len(old_vals) != 0:
                        deltas = new_vals - old_vals
                        deltas = np.insert(deltas, 0 , row["DX"])
                        deltas = np.insert(deltas, 0 , row["Month"])
                        deltas = np.insert(deltas, 0 , pat_id+ "-NR" + str(i))

                        df_data.append(deltas)

                        i = i+1

                    else:
                        continue

                last_sample_key = sample_key

        cols = df_thick.columns
        cols = np.insert(cols, 1, "DX")
        cols = np.insert(cols, 1, "Month")
        fin_df = pd.DataFrame(df_data, columns = cols)
        #for index, row in df.iterrows():


    def save_stats(self, path):
        self.get_dataframe().to_csv(path, index=False)

