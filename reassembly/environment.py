from dependencies import *

class Environment:
    def __init__(self, num: int, device=set_device()):
        self.device = device

        n = int(num ** (1 / 3))
        if n%2 != 0: n = n - 1
        
        x_points = torch.linspace(-n/2, n/2, n+1)
        y_points = torch.linspace(-n/2, n/2, n+1)
        z_points = torch.linspace(-n/2, n/2, n+1)
        
        xx, yy, zz = torch.meshgrid(x_points, y_points, z_points, indexing='ij')
        environment_points = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

        self.N = n/2
        self.environment = environment_points.to(self.device)
        self.intersection_points = torch.empty((0, self.environment.shape[1]), dtype=torch.float32, device=self.device)
        self.intersection_value = torch.tensor(0.0, dtype=torch.float32, device=self.device)

    def desc_environment(self):
        environment_points = self.environment.cpu().numpy()
        print('Environment points shape:', environment_points.shape, 'Environment points length:', len(environment_points))

    def get_intersections(self, meshFields: list, mode="counting"):
        switch_case = {
            "counting": self.get_intersections_counting,
            "multiplex": self.get_intersections_multiplex,
            # "counting_proximity": self.get_intersections_counting_proximity,
            # "multiplex_proximity": self.get_intersections_multiplex_proximity,
        }

        if mode in switch_case:
            return switch_case[mode](meshFields)
        else:
            raise ValueError(f"Invalid mode: {mode}. Valid modes are: {', '.join(switch_case.keys())}")


    def get_intersections_counting(self, meshFields: list):
        env_points_set = set(map(tuple, self.environment.cpu().numpy()))

        mesh_points_all = []
        mesh_values_all = []

        for mesh in meshFields:
            mesh.sdf_points = mesh.sdf_points.to(self.device)
            mesh_points_all.append(mesh.sdf_points)
            mesh_values_all.append(mesh.sdf_values)

        # Concatenate all mesh points and values
        all_mesh_points = torch.cat(mesh_points_all, dim=0)

        # Convert environment points to set for faster lookups
        env_points_set = set(map(tuple, self.environment.cpu().numpy()))
        
        # Find unique mesh points
        unique_mesh_points, unique_indices = torch.unique(all_mesh_points, dim=0, return_inverse=True)
        
        # Mask to find valid mesh points present in the environment
        valid_mask = torch.tensor([tuple(p) in env_points_set for p in unique_mesh_points.cpu().detach().numpy()], device=self.device)
        
        valid_mesh_points = unique_mesh_points[valid_mask]

        # Count occurrences of each valid mesh point
        point_counts = torch.zeros(valid_mesh_points.size(0), dtype=torch.int32, device=self.device)
        for mesh in meshFields:
            mesh_mask = torch.tensor([tuple(p) in env_points_set for p in mesh.sdf_points.cpu().detach().numpy()], device=self.device)
            mesh_points_valid = mesh.sdf_points[mesh_mask]
            idxs = torch.nonzero((valid_mesh_points.unsqueeze(1) == mesh_points_valid.unsqueeze(0)).all(dim=2), as_tuple=False)
            point_counts[idxs[:, 0]] += 1

        # Filter points that are intersected by more than one mesh
        multi_intersection_mask = point_counts > 1
        final_intersection_points = valid_mesh_points[multi_intersection_mask]

        self.intersection_points = final_intersection_points
        self.intersection_value = torch.tensor(final_intersection_points.size(0), dtype=torch.float32, device=self.device)

    def get_intersections_multiplex(self, meshFields: list):
        intersection_value = 0.0

        env_points_set = set(map(tuple, self.environment.cpu().numpy()))

        mesh_points_all = []
        mesh_values_all = []

        for mesh in meshFields:
            mesh.sdf_points = mesh.sdf_points.to(self.device)
            mesh_points_all.append(mesh.sdf_points)
            mesh_values_all.append(mesh.sdf_values)

        # Concatenate all mesh points and values
        all_mesh_points = torch.cat(mesh_points_all, dim=0)

        # Convert environment points to set for faster lookups
        env_points_set = set(map(tuple, self.environment.cpu().numpy()))
        
        # Find unique mesh points
        unique_mesh_points, unique_indices = torch.unique(all_mesh_points, dim=0, return_inverse=True)
        
        # Mask to find valid mesh points present in the environment
        valid_mask = torch.tensor([tuple(p) in env_points_set for p in unique_mesh_points.cpu().detach().numpy()], device=self.device)
        
        valid_mesh_points = unique_mesh_points[valid_mask]

        # Count occurrences of each valid mesh point
        point_counts = torch.zeros(valid_mesh_points.size(0), dtype=torch.int32, device=self.device)
        for mesh in meshFields:
            mesh_mask = torch.tensor([tuple(p) in env_points_set for p in mesh.sdf_points.cpu().detach().numpy()], device=self.device)
            mesh_points_valid = mesh.sdf_points[mesh_mask]
            idxs = torch.nonzero((valid_mesh_points.unsqueeze(1) == mesh_points_valid.unsqueeze(0)).all(dim=2), as_tuple=False)
            point_counts[idxs[:, 0]] += 1

        # Filter points that are intersected by more than one mesh
        multi_intersection_mask = point_counts > 1
        final_intersection_points = valid_mesh_points[multi_intersection_mask]

        # Compute intersection value
        for point in final_intersection_points:
            value = 0.0
            for mesh in meshFields:
                idx = torch.all(mesh.sdf_points == point, dim=1)
                value += mesh.sdf_values[idx].sum().item()
            intersection_value += value

        self.intersection_points = final_intersection_points
        self.intersection_value = torch.tensor(intersection_value, dtype=torch.float32, device=self.device)

    def get_intersections_counting_proximity(self, meshFields: list, tol=2):
        # Initialize tensors to collect intersection points
        intersection_counts = torch.zeros(self.environment.size(0), dtype=torch.int32, device=self.device)

        # Compare each mesh's points to the environment points
        for mesh in meshFields:
            mesh.sdf_points = mesh.sdf_points.to(self.device)

            # Expand both sets of points to form all pairs (environment x mesh)
            env_expanded = self.environment.unsqueeze(1)  # Shape: (num_env_points, 1, point_dims)
            mesh_expanded = mesh.sdf_points.unsqueeze(0)  # Shape: (1, num_mesh_points, point_dims)

            # Compute proximity based on the specified tolerance
            close = torch.abs(env_expanded - mesh_expanded) < tol  # Shape: (num_env_points, num_mesh_points, point_dims)
            is_intersection = torch.any(torch.all(close, dim=2), dim=1)  # Collapse to get points within tolerance

            # Update the intersection counts
            intersection_counts += is_intersection.int()

        # Identify points with intersections from multiple meshes
        valid_intersections = intersection_counts > 1

        # Gather the points and their intersection counts
        self.intersection_points = self.environment[valid_intersections]
        self.intersection_points = self.intersection_points.to(self.device)
        self.intersection_value = intersection_counts[valid_intersections].float().sum().to(self.device)

    def get_intersections_multiplex_proximity(self, meshFields: list, tol=2):
        intersection_points = torch.empty((0, self.environment.shape[1]), dtype=torch.float32, device=self.device)
        intersection_value = 0.0

        for point in self.environment:
            count = 0
            for mesh in meshFields:
                if torch.sum(torch.all(torch.abs(mesh.sdf_points - point) < tol, dim=1)) > 2:
                    count += 1
                if count > 1:
                    intersection_points = torch.cat((intersection_points, point.unsqueeze(0)), dim=0)
                    value = 0.0
                    for m in meshFields:
                        idx = torch.all(torch.abs(m.sdf_points - point) < tol, dim=1)
                        value += m.sdf_values[idx].sum().item()
                    intersection_value += value
                    break

        self.intersection_value = torch.tensor(intersection_value, dtype=torch.float32, device=self.device)
        self.intersection_points = intersection_points.to(self.device)

    

    



    



    

