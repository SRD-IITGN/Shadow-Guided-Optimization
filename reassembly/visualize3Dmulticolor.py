import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures.meshes import join_meshes_as_scene
import os
import plotly.graph_objects as go
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, help='Folder containing .obj files')
selected_folder = parser.parse_args().folder

def mesh_apply_colors(mesh, device):
    verts = mesh.verts_packed()
    num_verts = verts.shape[0]
    random_color = torch.rand((1, 3)).to(device)
    random_color = random_color.repeat(num_verts, 1)
    textures = TexturesVertex(verts_features=random_color.unsqueeze(0))
    mesh.textures = textures
    return mesh

def visualize_mesh(meshes, device):
    colored_meshes = [mesh_apply_colors(mesh, device) for mesh in meshes]
    combined_mesh = join_meshes_as_scene(colored_meshes)

    verts = combined_mesh.verts_packed().cpu().numpy()
    faces = combined_mesh.faces_packed().cpu().numpy()
    colors = combined_mesh.textures.verts_features_packed().cpu().numpy()

    # Normalize colors to range [0, 1]
    colors = colors.clip(0, 1)

    # Create Plotly mesh
    mesh = go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        vertexcolor=colors,
        name='Mesh'
    )

    fig = go.Figure(data=[mesh])
    fig.update_layout(scene_aspectmode='data')
    fig.show()

def main(selected_folder):
    meshes = []
    mesh_names = [f for f in os.listdir(selected_folder) if f.endswith('.obj')]
    for mesh_name in mesh_names:
        mesh = load_objs_as_meshes([os.path.join(selected_folder, mesh_name)], device=device)
        meshes.append(mesh)
    visualize_mesh(meshes, device)

main(selected_folder)
