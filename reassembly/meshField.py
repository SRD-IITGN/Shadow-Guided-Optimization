from dependencies import *

class meshField:
    def __init__(self, device=set_device()):     
        self.device = device   
        self.mesh = None
        self.sdf_points = None
        self.sdf_values = None

    def populate_Mesh(self, vertices, faces, scaling_factor=1.0, environment_factor=1.0):
        faces_idx = faces.verts_idx
        vertices = vertices
        center = vertices.mean(0)
        vertices = vertices - center
        scale = max(vertices.abs().max(0)[0])
        vertices = vertices / scale
        vertices = vertices*scaling_factor*environment_factor
        
        self.mesh = Meshes(
            verts=[vertices],
            faces=[faces_idx]
        ).to(self.device)

    def populate_Mesh_scale_consistent(self, vertices, faces, scaling_factor=1.0, environment_factor=1.0, consistency_factor=None):
        faces_idx = faces.verts_idx
        vertices = vertices
        center = vertices.mean(0)
        vertices = vertices - center

        vertices = vertices.to(self.device)
        faces_idx = faces_idx.to(self.device)

        if consistency_factor == None:
            scale = max(vertices.abs().max(0)[0])
        else:
            scale = consistency_factor
        vertices = vertices / scale
        vertices = vertices*scaling_factor*environment_factor
        self.mesh = Meshes(
            verts=[vertices],
            faces=[faces_idx]
        ).to(self.device)
        return scale
    
    def populate_textured_Mesh_scale_consistent(self, vertices, faces, scaling_factor=1.0, environment_factor=1.0, consistency_factor=None, texture=None):
        faces_idx = faces
        vertices = vertices
        center = vertices.mean(0)
        vertices = vertices - center

        vertices = vertices.to(self.device)
        faces_idx = faces_idx.to(self.device)

        if consistency_factor == None:
            scale = max(vertices.abs().max(0)[0]).to(self.device)
        else:
            scale = consistency_factor.to(self.device)
        vertices = vertices / scale
        vertices = vertices*scaling_factor*environment_factor
        self.mesh = Meshes(
            verts=[vertices],
            faces=[faces_idx],
            textures=texture
        ).to(self.device)
        return scale

    def populate_SDF(self, environment_points):
        self.sdf_points, self.sdf_values = self.get_SDF(environment_points=environment_points)

    def transform_mesh(self, rotate, translate):
        rotate = axis_angle_to_quaternion(rotate)
        self.mesh.verts_list()[0] = quaternion_apply(rotate, self.mesh.verts_list()[0])
        self.mesh.verts_list()[0] += translate

    def get_SDF(self, environment_points):
        verts, faces = self.mesh.get_mesh_verts_faces(0)[0].detach().cpu().numpy(), self.mesh.get_mesh_verts_faces(0)[1].detach().cpu().numpy()
        environment_points = environment_points.cpu().numpy()
        sdf = SDF(verts, faces)
        
        sdf_pts = []
        sdf_vals = []
        
        for point in environment_points:
            if sdf.contains(point):
                sdf_pts.append(point)
                sdf_vals.append(sdf(point))
        sdf_pts = np.array(sdf_pts)
        sdf_vals = np.array(sdf_vals)

        sdf_pts = torch.tensor(sdf_pts, dtype=torch.float32, device=self.device)
        sdf_vals = torch.tensor(sdf_vals, dtype=torch.float32, device=self.device)
        return sdf_pts, sdf_vals
    
    def transform_SDF(self, rotate, translate):
        rotate = axis_angle_to_quaternion(rotate)
        new_sdf_pts = quaternion_apply(rotate, self.sdf_points)
        new_sdf_pts += translate
        # round to zero decimal places
        new_sdf_pts = torch.round(new_sdf_pts)
        self.sdf_points = new_sdf_pts
        
    def clone(self):
        new_meshField = meshField()
        new_meshField.mesh = self.mesh.clone()
        new_meshField.sdf_points = self.sdf_points.clone()
        new_meshField.sdf_values = self.sdf_values.clone()
        return new_meshField