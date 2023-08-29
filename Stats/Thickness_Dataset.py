import nibabel.freesurfer.io as fsio
import numpy as np
import pandas as pd
import csv
import sys
from tqdm import tqdm

from Stats.Brain_Analyser import *
from Stats.Dataset_Loader import *

from pytorch3d.ops import (
    knn_points,
    knn_gather,
    sample_points_from_meshes
)

### Added some extra func. to use curvature as a metric as well
class Thickness_Dataset:
    def __init__(self, data_loader : Dataset_Loader, fs_avg_path : str, torchdevice : str = "cuda:0", measure : str = "thick"):
        self.brain_analyser = Brain_Analyser()
        self.data_loader = data_loader
        self.device = torch.device(torchdevice if torch.cuda.is_available() else "cpu")
        self.measure = measure
        self.data_lh = {}
        self.data_rh = {}
        self.fs_avg_sq_wmlh, self.fs_avg_sq_wmrh = self.load_fs_avg()

    def load_fs_avg(self):
        v_lh_wm, fc_lh_wm = fsio.read_geometry(fs_avg_path + "/" + "lh.sphere.reg")
        v_rh_wm, fc_rh_wm = fsio.read_geometry(fs_avg_path + "/" + "rh.sphere.reg")
        wm_lh = Meshes([torch.from_numpy(v_lh_wm.astype(np.float32)).to(self.device)],[torch.from_numpy(fc_lh_wm.astype(np.float32)).to(self.device)])
        wm_rh = Meshes([torch.from_numpy(v_rh_wm.astype(np.float32)).to(self.device)],[torch.from_numpy(fc_rh_wm.astype(np.float32)).to(self.device)])
        return wm_lh, wm_rh

    # This function generates all the thickness data fo the full dataset of the loader
    def generate_data(self, maximal_nr_of_elems = -1):
        for brain_dict in tqdm(self.data_loader):
            # Condition to break the loop
            if maximal_nr_of_elems != -1:
                maximal_nr_of_elems = maximal_nr_of_elems - 1
                if maximal_nr_of_elems == -1:
                    break

            idx_lh = self.knn_map(self.fs_avg_sq_wmlh, brain_dict['sq_reg_lh'])
            idx_rh = self.knn_map(self.fs_avg_sq_wmrh, brain_dict['sq_reg_rh'])
            if self.measure == "thick":
                self.data_lh[brain_dict['name']] = self.brain_analyser.calc_thickness_array(brain_dict['wm_lh'], brain_dict['pi_lh'])[idx_lh]
                self.data_rh[brain_dict['name']] = self.brain_analyser.calc_thickness_array(brain_dict['wm_rh'], brain_dict['pi_rh'])[idx_rh]
            elif self.measure == "curv":
                self.data_lh[brain_dict['name']] = self.brain_analyser.curv_from_cotcurv_laplacian(brain_dict['wm_lh'].verts_packed(), brain_dict['wm_lh'].faces_packed()).cpu().numpy()[idx_lh]
                self.data_rh[brain_dict['name']] = self.brain_analyser.curv_from_cotcurv_laplacian(brain_dict['wm_rh'].verts_packed(), brain_dict['wm_rh'].faces_packed()).cpu().numpy()[idx_rh]
        return

    def knn_map(self, mesh1, mesh2):
        _, knn_idx, _ = knn_points(mesh1.verts_packed().type(torch.float32)[None, :], mesh2.verts_packed().type(torch.float32)[None, :])
        return knn_idx.cpu().numpy().squeeze()

    def save_all_as_npy(self, path):
        
        try:
            os.mkdir(path + "/lh")
        except:
            print("LH dir already exists")
        
        try:
            os.mkdir(path + "/rh")
        except:
            print("RH dir already exists")
        
        for key in self.data_lh:
            with open(path + "/lh/" + str(key) + ".npy", "w"):
                np.save(path + "/lh/" + str(key), self.data_lh[key])
        for key in self.data_rh:
            with open(path + "/rh/" + str(key) + ".npy", "w"):
                np.save(path + "/rh/" + str(key), self.data_rh[key])   
