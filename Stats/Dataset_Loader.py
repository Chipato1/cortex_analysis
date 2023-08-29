import os
import trimesh
import numpy as np
import nibabel.freesurfer.io as fsio
import nibabel as nb
import torch
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.io import IO
from pytorch3d.ops import (
    knn_points,
    knn_gather,
    sample_points_from_meshes
)
# This should be an iterable dataloader
class Dataset_Loader:
    # Supported Type of models: PLY, FS_Orig
    # Registration always has to be in FS format
    def __init__(self, data_path, registration_path, model_type, map_registration = False, torchdevice = "cuda:0", only_numbers = True):
        self.model_type = model_type
        self.data_path = data_path
        self.registration_path = registration_path
        self.map_registration = map_registration
        self.only_numbers = only_numbers
        self.subdirs = self.get_immediate_subdirectories(data_path)
        self.device = torch.device(torchdevice if torch.cuda.is_available() else "cpu")

    def __iter__(self):
        for subdir in self.subdirs:
            yield self.load_mesh(subdir, self.model_type)
            
    def __len__(self):
        return len(self.subdirs)
    
        
    def __getitem__(self, idx):
        return self.load_mesh(self.subdirs[idx], self.model_type)

    #V2C: lh.aparc.DKTatlas40.annot
    #FS: lh.aparc.DKTatlas.annot
    def get_annotation(self):
        path = self.subdirs[0]

        if self.model_type == "PLY":
            #annot = fsio.read_annot(self.registration_path + "/" + path + "/label/lh.aparc.DKTatlas40.annot")
            annot = fsio.read_annot(self.registration_path + "/" + path + "/label/lh.aparc.annot")
        else:
            annot = fsio.read_annot(self.registration_path + "/" + path + "/label/rh.aparc.DKTatlas.annot")
        # 0 an -1 is 'unknown'
        annot[0][annot[0] == -1] = 0
        
        return annot

    # The two next classes are sister classes
    def get_annotation_classes(self):
        # Only return names that also appear as labels
        annot = self.get_annotation()
        return [annot[2][i] for i in set(annot[0])] 
        
    def get_sorted_available_indices(self):
        annot = self.get_annotation()
        return sorted(list(set(annot[0])))

    def load_single_brain_meshes_fs_clean(self, path):
        # Pipe for FreeSurfer
        
        v_lh_wm, fc_lh_wm = fsio.read_geometry(self.data_path + "/" + path + "/surf/" + "lh.white")
        v_rh_wm, fc_rh_wm = fsio.read_geometry(self.data_path + "/" + path + "/surf/" + "rh.white")
        v_lh_pi, fc_lh_pi = fsio.read_geometry(self.data_path + "/" + path + "/surf/" + "lh.pial")
        v_rh_pi, fc_rh_pi = fsio.read_geometry(self.data_path + "/" + path + "/surf/" + "rh.pial")
        
        wm_lh  = Meshes([torch.from_numpy(v_lh_wm.astype(np.float32)).to(self.device)],[torch.from_numpy(fc_lh_wm.astype(np.float32)).to(self.device)])
        wm_rh = Meshes([torch.from_numpy(v_rh_wm.astype(np.float32)).to(self.device)],[torch.from_numpy(fc_rh_wm.astype(np.float32)).to(self.device)])
        pi_lh = Meshes([torch.from_numpy(v_lh_pi.astype(np.float32)).to(self.device)],[torch.from_numpy(fc_lh_pi.astype(np.float32)).to(self.device)])
        pi_rh = Meshes([torch.from_numpy(v_rh_pi.astype(np.float32)).to(self.device)],[torch.from_numpy(fc_rh_pi.astype(np.float32)).to(self.device)])
        return {'name' : path, 'pi_lh': pi_lh, 'pi_rh': pi_rh, 'wm_lh': wm_lh, 'wm_rh': wm_rh}

    def load_single_brain_meshes_fs_orig(self, path):
        # Pipe for FreeSurfer
        
        v_lh_wm, fc_lh_wm = fsio.read_geometry(self.data_path + "/" + path + "/surf/" + "lh.white")
        v_rh_wm, fc_rh_wm = fsio.read_geometry(self.data_path + "/" + path + "/surf/" + "rh.white")
        v_lh_pi, fc_lh_pi = fsio.read_geometry(self.data_path + "/" + path + "/surf/" + "lh.pial.T1")
        v_rh_pi, fc_rh_pi = fsio.read_geometry(self.data_path + "/" + path + "/surf/" + "rh.pial.T1")
        
        wm_lh  = Meshes([torch.from_numpy(v_lh_wm.astype(np.float32)).to(self.device)],[torch.from_numpy(fc_lh_wm.astype(np.float32)).to(self.device)])
        wm_rh = Meshes([torch.from_numpy(v_rh_wm.astype(np.float32)).to(self.device)],[torch.from_numpy(fc_rh_wm.astype(np.float32)).to(self.device)])
        pi_lh = Meshes([torch.from_numpy(v_lh_pi.astype(np.float32)).to(self.device)],[torch.from_numpy(fc_lh_pi.astype(np.float32)).to(self.device)])
        pi_rh = Meshes([torch.from_numpy(v_rh_pi.astype(np.float32)).to(self.device)],[torch.from_numpy(fc_rh_pi.astype(np.float32)).to(self.device)])

        return {'name' : path, 'pi_lh': pi_lh, 'pi_rh': pi_rh, 'wm_lh': wm_lh, 'wm_rh': wm_rh}
    
    def load_single_brain_meshes_obj(self, path):
        # Pipe for ply based stuff
        pi_lh = IO().load_mesh(self.data_path + "/" + path + "/gm_adni_lh.obj", device = self.device)
        pi_rh = IO().load_mesh(self.data_path + "/" + path + "/gm_adni_rh.obj", device = self.device)
        wm_lh = IO().load_mesh(self.data_path + "/" + path + "/wm_adni_lh.obj", device = self.device)
        wm_rh = IO().load_mesh(self.data_path + "/" + path + "/wm_adni_rh.obj", device = self.device)
        return {'name' : path, 'pi_lh': pi_lh, 'pi_rh': pi_rh, 'wm_lh': wm_lh, 'wm_rh': wm_rh}
    
    def load_single_brain_meshes_ply_csr(self, path):
        # Pipe for ply based stuff
        pi_lh = IO().load_mesh(self.data_path + "/" + path + "/pred_lh_pial_0.5_top_fix.ply", device = self.device)
        pi_rh = IO().load_mesh(self.data_path + "/" + path + "/pred_rh_pial_0.5_top_fix.ply", device = self.device)
        wm_lh = IO().load_mesh(self.data_path + "/" + path + "/pred_lh_white_0.5_top_fix.ply", device = self.device)
        wm_rh = IO().load_mesh(self.data_path + "/" + path + "/pred_rh_white_0.5_top_fix.ply", device = self.device)
        return {'name' : path, 'pi_lh': pi_lh, 'pi_rh': pi_rh, 'wm_lh': wm_lh, 'wm_rh': wm_rh}
    
    def load_single_brain_meshes_ply(self, path):
        # Pipe for ply based stuff
        pi_lh = IO().load_mesh(self.data_path + "/" + path + "/lh_pial.ply", device = self.device)
        pi_rh = IO().load_mesh(self.data_path + "/" + path + "/rh_pial.ply", device = self.device)
        wm_lh = IO().load_mesh(self.data_path + "/" + path + "/lh_white.ply", device = self.device)
        wm_rh = IO().load_mesh(self.data_path + "/" + path + "/rh_white.ply", device = self.device)
        return {'name' : path, 'pi_lh': pi_lh, 'pi_rh': pi_rh, 'wm_lh': wm_lh, 'wm_rh': wm_rh}

    def load_single_brain_meshes_gifti(self, path):
        c_path = self.data_path + "/" + path + "/"
        v_lh_pi, fc_lh_pi = nb.load(c_path + "lh.pial.orig_out.gii.gz").darrays[0].data, nb.load(c_path + "lh.pial.orig_out.gii.gz").darrays[1].data
        v_rh_pi, fc_rh_pi = nb.load(c_path + "rh.pial.orig_out.gii.gz").darrays[0].data, nb.load(c_path + "rh.pial.orig_out.gii.gz").darrays[1].data
        v_lh_wm, fc_lh_wm = nb.load(c_path + "lh.white.orig_out.gii.gz").darrays[0].data, nb.load(c_path + "lh.white.orig_out.gii.gz").darrays[1].data
        v_rh_wm, fc_rh_wm = nb.load(c_path + "rh.white.orig_out.gii.gz").darrays[0].data, nb.load(c_path + "rh.white.orig_out.gii.gz").darrays[1].data

        wm_lh  = Meshes([torch.from_numpy(v_lh_wm.astype(np.float32)).to(self.device)],[torch.from_numpy(fc_lh_wm.astype(np.float32)).to(self.device)])
        wm_rh = Meshes([torch.from_numpy(v_rh_wm.astype(np.float32)).to(self.device)],[torch.from_numpy(fc_rh_wm.astype(np.float32)).to(self.device)])
        pi_lh = Meshes([torch.from_numpy(v_lh_pi.astype(np.float32)).to(self.device)],[torch.from_numpy(fc_lh_pi.astype(np.float32)).to(self.device)])
        pi_rh = Meshes([torch.from_numpy(v_rh_pi.astype(np.float32)).to(self.device)],[torch.from_numpy(fc_rh_pi.astype(np.float32)).to(self.device)])
        return {'name' : path, 'pi_lh': pi_lh, 'pi_rh': pi_rh, 'wm_lh': wm_lh, 'wm_rh': wm_rh}

    def get_partner_dict(self, path, brain_dict, brain_half):
        fs_reg_brain_str = ""
        gt_brain = None
        
        if brain_half == "lh":
            fs_reg_brain_str = "lh.white"
            gt_brain_pts = brain_dict["wm_lh"].verts_packed().type(torch.float32)[None, :]
        else:
            fs_reg_brain_str = "rh.white"
            gt_brain_pts = brain_dict["wm_rh"].verts_packed().type(torch.float32)[None, :]
        
        verts, _ = fsio.read_geometry(self.registration_path + "/" + path + "/surf/" + fs_reg_brain_str)
        verts = torch.from_numpy(verts).to(self.device).type(torch.float32)[None, :]
        
        _, knn_idx, _ = knn_points(gt_brain_pts, verts)
        
        return knn_idx.cpu().numpy().squeeze()

    # Proxy method 
    def load_mesh(self, path, method):
        output = None
        if method == "PLY" or method == "PLY_CF":
            output = self.load_single_brain_meshes_ply(path)
        elif method == "OBJ":
            output = self.load_single_brain_meshes_obj(path)    
        elif method == "FS_ORIG":
            output = self.load_single_brain_meshes_fs_orig(path)
        elif method == "FS_CLEAN":
            output = self.load_single_brain_meshes_fs_clean(path)
        elif method == "GIFTI":
            output = self.load_single_brain_meshes_gifti(path)
        elif method == "PLY_CSR":
            output = self.load_single_brain_meshes_ply_csr(path)
        else:
            raise Exception("Sorry, Datatype not supported!")

        if self.registration_path != None:
            v_lh_wm, fc_lh_wm = fsio.read_geometry(self.registration_path + "/" + path + "/surf/" + "lh.sphere.reg")
            v_rh_wm, fc_rh_wm = fsio.read_geometry(self.registration_path + "/" + path + "/surf/" + "rh.sphere.reg")
            output["sq_reg_lh"] = Meshes([torch.from_numpy(v_lh_wm.astype(np.float32)).to(self.device)],[torch.from_numpy(fc_lh_wm.astype(np.float32)).to(self.device)])
            output["sq_reg_rh"] = Meshes([torch.from_numpy(v_rh_wm.astype(np.float32)).to(self.device)],[torch.from_numpy(fc_rh_wm.astype(np.float32)).to(self.device)])
            
            if method == "OBJ" or method == "PLY_CF":
                output["an_rh"] = fsio.read_annot(self.registration_path + "/" + path + "/label/rh.aparc.annot")[0]
                output["an_lh"] = fsio.read_annot(self.registration_path + "/" + path + "/label/lh.aparc.annot")[0]

            if method == "PLY" or  method == "FS_CLEAN" or method == "PLY_CSR":
                output["an_rh"] = fsio.read_annot(self.registration_path + "/" + path + "/label/rh.aparc.annot")[0]
                output["an_lh"] = fsio.read_annot(self.registration_path + "/" + path + "/label/lh.aparc.annot")[0]

            if  method == "FS_ORIG":
                output["an_rh"] = fsio.read_annot(self.registration_path + "/" + path + "/label/rh.aparc.DKTatlas.annot")[0]
                output["an_lh"] = fsio.read_annot(self.registration_path + "/" + path + "/label/lh.aparc.DKTatlas.annot")[0]
            # Map -1 to 0 because both is unkown for us
            output["an_rh"][output["an_rh"] == -1] = 0
            output["an_lh"][output["an_lh"] == -1] = 0
            
        if self.map_registration == True:
            output["lh_reg_map"] = self.get_partner_dict(path, output, "lh")
            output["rh_reg_map"] = self.get_partner_dict(path, output, "rh")

        return output

    #Helpers
    # This function helps to prevent loading folders that are not subjects. Subjects-Folders have to be all numbers, be a directory and also contain some stuff
    def get_immediate_subdirectories(self, a_dir):
        return [name for name in os.listdir(a_dir)
                if os.path.isdir(os.path.join(a_dir, name)) and len(os.listdir(os.path.join(a_dir, name))) != 0 and (name.isdigit() or self.only_numbers == False)]
