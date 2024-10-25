from dependencies import *

class Rendering:
    def __init__(self, camera_transform:list, camera_type:str='orthographic', image_size:int=128, sigma:float = 1e-4, faces_per_pixel:int=50, environment_factor:int=1.0, device=set_device(), mode=None):
        self.device = device
        self.camera = self.set_camera(camera_transform, camera_type=camera_type, environment_factor=environment_factor).to(self.device)
        self.raster_settings = self.get_raster_settings(image_size=image_size, sigma=sigma, faces_per_pixel=faces_per_pixel)
        if mode == 'silhouette':
            self.renderer = self.silhoutte_renderer()
        elif mode == 'texture':
            self.renderer = self.texture_renderer()        

    def set_camera(self, camera_transform:list, camera_type, environment_factor:int):
        if camera_type == 'orthographic':
            R, T = look_at_view_transform(dist=environment_factor, elev=camera_transform[1], azim=camera_transform[2])
            camera = FoVOrthographicCameras(R=R, T=T, min_x=-1*environment_factor, max_x=1*environment_factor, min_y=-1*environment_factor, max_y=1*environment_factor)
        elif camera_type == 'perspective':
            R, T = look_at_view_transform(dist=environment_factor*2, elev=camera_transform[1], azim=camera_transform[2])
            camera = FoVPerspectiveCameras(R=R, T=T, fov=90, degrees=True)
        return camera

    def get_raster_settings(self, image_size, sigma, faces_per_pixel):
        raster_settings_soft = RasterizationSettings(
            image_size=image_size,
            blur_radius=np.log(1. / 1e-4 - 1.)*sigma,
            faces_per_pixel=faces_per_pixel,
            bin_size=None
        )
        return raster_settings_soft

    def silhoutte_renderer(self):
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.camera,
                raster_settings=self.raster_settings
            ),
            shader=SoftSilhouetteShader()
        )
        return renderer.to(self.device)
    
    def texture_renderer(self):
        # get the camera position
        camera_position = self.camera.get_camera_center()
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.camera, 
                raster_settings=self.raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device, 
                cameras=self.camera,
                lights=PointLights(device=self.device, location=camera_position)
            )
        )
        return renderer.to(self.device)
    
    def project_silhouette(self, meshes:list):
        combined_meshes = join_meshes_as_scene(meshes)
        images = self.renderer(combined_meshes)
        silhouette = images[0][..., 3]
        return silhouette

    def project_texture(self, meshes:list):
        combined_meshes = join_meshes_as_scene(meshes)
        images = self.renderer(combined_meshes)
        silhouette = images[0][..., :3]
        return silhouette