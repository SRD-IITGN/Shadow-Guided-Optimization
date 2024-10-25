from dependencies import *
import torch.nn.functional as F
import torch.nn as nn
from pytorch3d.loss import (
    chamfer_distance, 
)

class grad_image(nn.Module):
    def __init__(self, gpu_id):
        super(grad_image, self).__init__()

        self.sobel_x_kernel = torch.tensor([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]], dtype=torch.float32).cuda(gpu_id)

        self.sobel_y_kernel = torch.tensor([[-1, -2, -1],
                                       [0, 0, 0],
                                       [1, 2, 1]], dtype=torch.float32).cuda(gpu_id)

    def forward(self, image_tensor):
        n_channels = image_tensor.shape[1]
        gradient_x = torch.nn.functional.conv2d(image_tensor, self.sobel_x_kernel.unsqueeze(0).unsqueeze(0).repeat(1,n_channels,1,1), padding=1)
        gradient_y = torch.nn.functional.conv2d(image_tensor, self.sobel_y_kernel.unsqueeze(0).unsqueeze(0).repeat(1,n_channels,1,1), padding=1)    
        gradient_magnitude = torch.sqrt(torch.relu(gradient_x**2 + gradient_y**2))
        return gradient_magnitude

class Losses:
    def get_intersection_loss(environment, mesh_fields, mode):
        environment.get_intersections(mesh_fields, mode)
        loss = environment.intersection_value
        return loss
    
    def get_distance_loss(mesh_fields):
        loss = 0.0
        for mesh in mesh_fields:
            loss += abs(mesh.mesh.get_mesh_verts_faces(0)[0].sum())
        return loss
    
    def get_environment_boundings_loss(mesh_fields, environment):
        loss = 0.0
        bound = environment.N
        for mesh in mesh_fields:
            # get all the points that are outside the environment
            loss += torch.sum(torch.abs(mesh.sdf_points) > bound).float()
        # exponentiate the loss
        # loss = 2**loss - 1
        return loss

    def get_LoG_loss(renderer, mesh_fields, target_silhouette, sigma=1):
        silhouette = renderer.project_silhouette([mesh.mesh for mesh in mesh_fields])
        grad_image_module = grad_image(0)
        silhouette = grad_image_module(silhouette.unsqueeze(0).unsqueeze(0))
        target_silhouette = grad_image_module(target_silhouette.unsqueeze(0).unsqueeze(0))
        loss = ((silhouette - target_silhouette)**2).mean()
        return loss
    
    def get_LoG_loss_texture(renderer, mesh_fields, target_silhouette):
        silhouette = renderer.project_texture([mesh.mesh for mesh in mesh_fields])
        grad_image_module = grad_image(0)
        silhouette = grad_image_module(silhouette)
        target_silhouette = grad_image_module(target_silhouette)
        loss = ((silhouette - target_silhouette)**2).mean()
        return loss

    def get_silhouette_loss(renderer, mesh_fields, target_silhouette):
        silhouette = renderer.project_silhouette([mesh.mesh for mesh in mesh_fields])
        loss = ((silhouette - target_silhouette)**2).mean() 
        return loss
    
    def get_texture_loss(renderer, mesh_fields, target_texture):
        texture = renderer.project_texture([mesh.mesh for mesh in mesh_fields])
        loss = F.l1_loss(texture, target_texture)
        return loss
    
    def get_silhouette_aware_texture_loss(renderer, mesh_fields, target_silhouette, target_texture):
        texture = renderer.project_texture([mesh.mesh for mesh in mesh_fields])
        silhouette = renderer.project_silhouette([mesh.mesh for mesh in mesh_fields])
        texture_clipped = texture * silhouette.unsqueeze(-1)
        target_clipped = target_texture * target_silhouette.unsqueeze(-1)
        loss = ((texture_clipped - target_clipped)**2).mean() * 10
        return loss

    def get_pixel_distance_loss(renderer, mesh_fields, target_texture):
        texture = renderer.project_texture([mesh.mesh for mesh in mesh_fields])
        loss = chamfer_distance(texture, target_texture)
        return loss[0]


        
      