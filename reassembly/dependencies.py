from tqdm import tqdm
import os
import sys
import torch
import pytorch3d
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    FoVOrthographicCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
    SoftSilhouetteShader
)
import matplotlib.pyplot as plt
import numpy as np
from pytorch3d.vis.plotly_vis import plot_scene
from pytorch3d.structures.meshes import Meshes, join_meshes_as_scene
import math
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from pysdf import SDF
import plotly.graph_objects as go
from pytorch3d.transforms import quaternion_apply, euler_angles_to_matrix, axis_angle_to_quaternion

def set_device(id=0):
    if id == 'cpu':
        device = torch.device("cpu")
        print("Using device: cpu")
        return device
    device_str = "cuda:{}".format(id)
    if torch.cuda.is_available():
        device = torch.device(device_str)
        torch.cuda.set_device(device)
        print("Using device: {}".format(device_str))
    else:
        device = torch.device("cpu")
        print("Using device: cpu")
    return device