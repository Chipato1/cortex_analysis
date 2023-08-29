import os
import nibabel.freesurfer.io as fsio
import numpy as np

from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss.point_mesh_distance import _PointFaceDistance
from pytorch3d.ops import cot_laplacian

import torch
import math

class Brain_Analyser: 

    def __init__(self):
        self.point_face_distance = _PointFaceDistance.apply

    def _point_mesh_face_distance_unidirectional(self, white_pntcloud: Pointclouds,
                                                pial_mesh: Meshes):
        """ Compute the cortical thickness for every point in 'white_pntcloud' as
        its distance to the surface defined by 'pial_mesh'."""
        # The following is taken from pytorch3d.loss.point_to_mesh_distance
        # Packed representation for white matter pointclouds
        points = white_pntcloud.points_packed()  # (P, 3)
        points_first_idx = white_pntcloud.cloud_to_packed_first_idx()
        max_points = white_pntcloud.num_points_per_cloud().max().item()

        # Packed representation for faces
        verts_packed = pial_mesh.verts_packed()
        faces_packed = pial_mesh.faces_packed()
        tris = verts_packed[faces_packed]  # (T, 3, 3)
        tris_first_idx = pial_mesh.mesh_to_faces_packed_first_idx()

        # Point to face distance: shape # (P,)
        point_to_face = self.point_face_distance(
            points.float(), points_first_idx, tris.float(), tris_first_idx, max_points
        )
        # Take root as point_face_distance returns squared distances
        point_to_face = torch.sqrt(point_to_face)

        return point_to_face
    
    def curv_from_cotcurv_laplacian(self, verts_packed, faces_packed):
        """ Construct the cotangent curvature Laplacian as done in
        pytorch3d.loss.mesh_laplacian_smoothing and use it for approximation of the
        mean curvature at each vertex. See also
        - Nealen et al. "Laplacian Mesh Optimization", 2006
        """
        # No backprop through the computation of the Laplacian (taken as a
        # constant), similar to pytorch3d.loss.mesh_laplacian_smoothing
        with torch.no_grad():
            L, inv_areas = cot_laplacian(verts_packed, faces_packed)
            L_sum = torch.sparse.sum(L, dim=1).to_dense().view(-1,1)
            norm_w = 0.25 * inv_areas

        return torch.norm(
            (L.mm(verts_packed) - L_sum * verts_packed) * norm_w,
            dim=1
        )
    
    # To methods are supported
    def calc_thickness_array(self,wm_surf, pial_surf):
        return self._point_mesh_face_distance_unidirectional(Pointclouds(wm_surf.verts_list()), pial_surf).cpu().numpy()