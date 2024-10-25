from dependencies import *
import imageio
from pytorch3d.io import save_obj

class Visualizer:
    def visualize_silhouette(self, silhouette, name:str='silhouette'):
        plt.imshow(silhouette.squeeze().cpu().numpy())
        plt.axis('off')
        plt.imsave(f'{name}.png', silhouette.squeeze().cpu().numpy())

    def visualize_silhouette_multiple(self, silhouettes, name: str = 'silhouettes'):
        silhouette_imgs = [silhouette.squeeze().cpu().numpy() for silhouette in silhouettes]
        n_silhouettes = len(silhouettes)
        height, width = silhouettes[0].shape[1], silhouettes[0].shape[0]
        aspect_ratio = width / height
        fig_width = min(100, n_silhouettes * aspect_ratio * 4)  
        fig_height = min(100, 4)  
        fig, axs = plt.subplots(1, n_silhouettes, figsize=(fig_width, fig_height))
        if n_silhouettes == 1:
            axs = [axs]
        for i, silhouette in enumerate(silhouette_imgs):
            axs[i].imshow(silhouette)
            axs[i].axis('off')
            axs[i].title.set_text(f'Camera {i+1}')
        plt.savefig(f'{name}.png')
        plt.close()

    def visualize_silhouette_multiple_black(self, silhouettes, name: str = 'silhouettes', dpi: int = 300):
        silhouette_imgs = [silhouette.squeeze().cpu().numpy() for silhouette in silhouettes]
        n_silhouettes = len(silhouettes)
        height, width = silhouettes[0].shape[1], silhouettes[0].shape[0]
        aspect_ratio = width / height
        fig_width = min(100, n_silhouettes * aspect_ratio * 4) 
        fig_height = min(100, 4)
        fig, axs = plt.subplots(1, n_silhouettes, figsize=(fig_width, fig_height), dpi=dpi)
        fig.patch.set_facecolor('black') 
        if n_silhouettes == 1:
            axs = [axs]
        for i, silhouette in enumerate(silhouette_imgs):
            axs[i].imshow(silhouette, cmap='gray', interpolation='nearest')
            axs[i].axis('off')
            axs[i].set_facecolor('black')  
        plt.subplots_adjust(wspace=0.1, hspace=0.1) 
        plt.savefig(f'{name}.png', bbox_inches='tight', pad_inches=0.1, facecolor='black')
        plt.close()

    def visualize_3d(self, meshes, name='scene'):
        combined_meshes = join_meshes_as_scene(meshes)
        fig = plot_scene({
            name: {
                "mesh": combined_meshes
            }
        })
        fig.write_html(f"{name}.html")

    def visualize_environment(self, environment, name:str='environment', env_color='gray', intersection_color='red'):
        environment_points = environment.environment.cpu().detach().numpy()
        fig = go.Figure(data=[go.Scatter3d(
            x=environment_points[:, 0],
            y=environment_points[:, 1],
            z=environment_points[:, 2],
            mode='markers',
            marker=dict(
                size=1,
                color=env_color,
                opacity=0.2
            )
        )])
        intersection_points = environment.intersection_points.cpu().detach().numpy()
        fig.add_trace(go.Scatter3d(
            x=intersection_points[:, 0],
            y=intersection_points[:, 1],
            z=intersection_points[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=intersection_color,
                opacity=1
            )
        ))
        fig.update_layout(scene=dict(
                            xaxis_title='X',
                            yaxis_title='Y',
                            zaxis_title='Z'),
                            margin=dict(l=0, r=0, b=0, t=0))
        # save the plot
        fig.write_html(name + '.html')

    def visualize_SDF(self, sdf_pts:list, sdf_vals:list, environment_points, name:str='SDF'):
        sdf_pts = [instance.cpu().numpy() for instance in sdf_pts]
        sdf_vals = [instance.cpu().numpy() for instance in sdf_vals]
        environment_points = environment_points.cpu().numpy()

        fig = go.Figure(data=[go.Scatter3d(
            x=environment_points[:, 0],
            y=environment_points[:, 1],
            z=environment_points[:, 2],
            mode='markers',
            marker=dict(
                size=1,
                color='gray',
                opacity=0.2
            )
        )])

        if len(sdf_pts[0]) > 1:
            for instance in range(len(sdf_pts)):
                fig.add_trace(go.Scatter3d(
                    x=sdf_pts[instance][:, 0],
                    y=sdf_pts[instance][:, 1],
                    z=sdf_pts[instance][:, 2],
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=sdf_vals[instance].flatten(),
                        colorscale='viridis',
                        opacity=1
                    )
                ))
                fig.update_layout(scene=dict(
                                    xaxis_title='X',
                                    yaxis_title='Y',
                                    zaxis_title='Z'),
                                    margin=dict(l=0, r=0, b=0, t=0))
    
        fig.write_html(name + '.html')

    def visualize_curves(self, curves, name:str='loss'):
        for curve in curves:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=curve["values"], mode='lines', name=curve["name"]))
            fig.update_layout(xaxis_title='Iterations', yaxis_title='Loss', title='Losses')
            fig.write_html(name + '_' + curve["name"] + '.html')  

    def save_obj(self, meshes, name='scene'):
        combined_meshes = join_meshes_as_scene(meshes)
        verts = combined_meshes.verts_list()[0]  # Getting vertices of the first (and only) mesh in the batch
        faces = combined_meshes.faces_list()[0]  # Getting faces of the first (and only) mesh in the batch
        save_obj(name+'.obj', verts, faces)

    def save_gif(self, image_folder, gif_name):
        images = []
        for file_name in sorted(os.listdir(image_folder)):
            if file_name.endswith(".png"):
                file_path = os.path.join(image_folder, file_name)
                images.append(imageio.imread(file_path))

        imageio.mimsave(gif_name, images, duration=0.5)
        print(f"Created gif {gif_name}")

    def save_indivisual_objs(self, meshes, name='scene'):
        os.makedirs(name, exist_ok=True)
        for i, mesh in enumerate(meshes):
            verts = mesh.verts_list()[0]  
            faces = mesh.faces_list()[0] 
            save_obj(name+'/'+str(i).zfill(10)+'.obj', verts, faces)
    
    def visualize_image_silhouette_multiple(self, images, name="scene"):
        # Check if the images are CUDA tensors and move them to CPU
        images = [image.cpu().numpy() if hasattr(image, 'is_cuda') and image.is_cuda else image for image in images]
        
        # Handle single image case by ensuring axs is iterable
        if len(images) == 1:
            fig, axs = plt.subplots(1, 1, figsize=(4, 4))
            axs.imshow(images[0])
            axs.axis('off')
        else:
            fig, axs = plt.subplots(1, len(images), figsize=(4*len(images), 4))
            for i, image in enumerate(images):
                axs[i].imshow(image)
                axs[i].axis('off')
        
        # Save figure to file
        plt.savefig(f'{name}.png')
        plt.close()