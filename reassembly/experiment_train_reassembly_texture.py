import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch3d")
import sys
import argparse
import importlib
import os
import shutil
from dependencies import *
from meshField import meshField
from environment import Environment
from visualizer import Visualizer
from rendering import Rendering
from losses import *
from PIL import Image

parser = argparse.ArgumentParser(description='parse command-line arguments.')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--save_mode', type=str, help='save mode [new/override]')
parser.add_argument('--device', type=str, help='device [cpu/id]')

ARG_config = parser.parse_args().config
ARG_save_mode = parser.parse_args().save_mode
hyperparameters = importlib.import_module(ARG_config)
ARG_device = parser.parse_args().device

# validate the arguments
assert ARG_save_mode in ['new', 'override'], "Invalid save mode. Choose from [new, override]"
# assert ARG_task in ['packing', 'reassembly'], "Invalid task. Choose from [packing, reassembly]"
assert ARG_device.isdigit() or ARG_device == 'cpu', "Invalid device. Choose from [cpu, id]"

# folder setup for saving the results
_folder = os.listdir('experiments')
_num = 0
for i in _folder:
    _n = i.split('_')[-1]
    if _n.isdigit():_num = max(_num, int(_n))

if ARG_save_mode == 'new':
    RUNS_FOLDER = 'experiments/runs_'+str(_num+1).zfill(4)
elif ARG_save_mode == 'override' and _num > 0:
    RUNS_FOLDER = 'experiments/runs_'+str(_num).zfill(4)

if os.path.exists(RUNS_FOLDER):shutil.rmtree(RUNS_FOLDER), os.makedirs(RUNS_FOLDER)
else: os.makedirs(RUNS_FOLDER)

os.mkdir(os.path.join(RUNS_FOLDER, 'silhouettes'))
os.mkdir(os.path.join(RUNS_FOLDER, 'metadata'))
os.mkdir(os.path.join(RUNS_FOLDER, 'report'))
os.mkdir(os.path.join(RUNS_FOLDER, 'report/structures'))

# copy the hyperparameters file to the results folder
hyperparam_file_path = os.path.join('config', ARG_config.split('.')[-1] + '.py')
shutil.copy(hyperparam_file_path, RUNS_FOLDER+'/metadata/params.txt')

# set the device
device = set_device(int(ARG_device))
# get the visualizer
visualizer = Visualizer()

# create the environment
environment = Environment(hyperparameters.ENVIRONMENT_SIZE, device=device)

# create the renderers
renderers = []
assert len(hyperparameters.CAMERA_TYPES) == hyperparameters.CAMERA_COUNT, "Number of camera transforms should be equal to the number of cameras"
for i in range(hyperparameters.CAMERA_COUNT):
        camera_transform = hyperparameters.CAMERA_TRANSFORMS[i] 
        renderer = Rendering(camera_transform, 
                             camera_type=hyperparameters.CAMERA_TYPES[i], 
                             image_size=hyperparameters.IMAGE_SIZE, 
                             sigma=1e-4, 
                             faces_per_pixel=hyperparameters.FACES_PER_PIXEL, 
                             environment_factor=environment.N,
                             mode='texture',
                             device=device)
        rendererSil = Rendering(camera_transform, 
                             camera_type=hyperparameters.CAMERA_TYPES[i], 
                             image_size=hyperparameters.IMAGE_SIZE, 
                             sigma=1e-4, 
                             faces_per_pixel=hyperparameters.FACES_PER_PIXEL, 
                             environment_factor=environment.N,
                             mode='silhouette',
                             device=device)
        renderers.append(renderer)

target_silhouettes = []
target_bnw = []
target_objs = []
OBJECT_TEXTURES = []
for trg_ in hyperparameters.TARGET_OBJECTS:
    vertices, faces, _ = load_obj(trg_)
    cols = torch.rand((1, 3)).to(device)
    cols = cols.repeat(vertices.shape[0], 1)
    textures = TexturesVertex(verts_features=cols.unsqueeze(0))
    t_mesh = Meshes(verts=[vertices], faces=[faces.verts_idx], textures=textures).to(device)
    target_objs.append(t_mesh)
    OBJECT_TEXTURES.append(textures)

target_obj_comb = join_meshes_as_scene(target_objs)

for i in range(hyperparameters.CAMERA_COUNT):
    target_object = target_obj_comb
    renderer = renderers[i]
    vertices, faces = target_object.verts_list()[0], target_object.faces_list()[0]
    target_mesh = meshField(device=device)
    texture = target_object.textures
    consistency_scale = target_mesh.populate_textured_Mesh_scale_consistent(vertices, faces, 
                                                                scaling_factor=hyperparameters.TARGET_SCALE, 
                                                                environment_factor=environment.N, consistency_factor=None, texture=texture)
    
    __targetTex__ = renderer.project_texture([target_mesh.mesh]).to(device)
    __targetBnW__ = renderer.project_silhouette([target_mesh.mesh]).to(device)
    target_silhouettes.append(__targetTex__)
    target_bnw.append(__targetBnW__)

visualizer.visualize_silhouette_multiple(target_silhouettes, name=RUNS_FOLDER+'/metadata/target_silhouettes')

# create the source objetcs
# source_objects = hyperparameters.SOURCE_OBJECTS
# __sourceMeshFields__ = []
# for idx, obj_path in enumerate(source_objects):
#     vertices, faces, _ = load_obj(obj_path)
#     mesh = meshField(device=device)

#     mesh.populate_textured_Mesh_scale_consistent(vertices, faces.verts_idx, scaling_factor=hyperparameters.SOURCE_SCALE, 
#                                         environment_factor=environment.N, consistency_factor=consistency_scale, texture=OBJECT_TEXTURES[idx])
    
#     mesh.populate_SDF(environment.environment)
    
#     # perform random transformations
#     if hyperparameters.RANDOM_INITIAL_TRANSFORMATIONS:
#         random_translation = (torch.rand(3, device=device) - 0.5) * environment.N 
#         random_rotation = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32) 
#         mesh.transform_mesh(rotate=random_rotation, translate=random_translation)
#         mesh.transform_SDF(rotate=random_rotation, translate=random_translation)

#     __sourceMeshFields__.append(mesh)

_src_mesh = target_obj_comb
verts, faces = _src_mesh.verts_list()[0], _src_mesh.faces_list()[0]
_mesh = meshField(device=device)
_mesh.populate_textured_Mesh_scale_consistent(verts, faces, scaling_factor=hyperparameters.SOURCE_SCALE,
                                            environment_factor=environment.N, consistency_factor=None, texture=target_obj_comb.textures)

_mesh.populate_SDF(environment.environment)
__sourceMeshFields__ = [_mesh]
random_translation = (torch.rand(3, device=device) - 0.5) * environment.N 
random_rotation = torch.tensor([5.0, 0.0, 3.0], dtype=torch.float32) 
__sourceMeshFields__[0].transform_mesh(rotate=random_rotation, translate=random_translation)
__sourceMeshFields__[0].transform_SDF(rotate=random_rotation, translate=random_translation)

source_silhouettes = []
for i in range(hyperparameters.CAMERA_COUNT):
    renderer = renderers[i]
    silhouette = renderer.project_texture([mesh.mesh for mesh in __sourceMeshFields__]) # project the source objects
    source_silhouettes.append(silhouette)
    visualizer.visualize_3d([i.mesh for i in __sourceMeshFields__], name=RUNS_FOLDER+'/metadata/source_objects_'+str(i)) # visualize the source objects in 3D
    visualizer.visualize_SDF([mesh.sdf_points for mesh in __sourceMeshFields__],
                            [mesh.sdf_values for mesh in __sourceMeshFields__],
                            environment.environment, name=RUNS_FOLDER+'/metadata/source_SDF_'+str(i))

visualizer.visualize_silhouette_multiple(source_silhouettes, name=RUNS_FOLDER+'/metadata/source_silhouettes') 

environment.get_intersections(__sourceMeshFields__) # get the intersections
visualizer.visualize_environment(environment, name=RUNS_FOLDER+'/metadata/environment_source') # visualize the environment with the objects

# setup weights for training
losses = {  
            "texture"    : {"weight": hyperparameters.WEIGHT_SILHOUETTE, "values": []},
            "intersection"  : {"weight": hyperparameters.WEIGHT_INTERSECTION, "values": []},   
            "bounding"      : {"weight": hyperparameters.WEIGHT_BOUNDING, "values": []},
            # "LoG"           : {"weight": hyperparameters.WEIGHT_LOG, "values": []}
         }

# setup the transformations tensors
transformations= []
for i in range(1):
    transformations.append(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device, requires_grad=True)) # for rotation
    transformations.append(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device, requires_grad=True)) # for translation

# setup the optimizer
optimizer = torch.optim.Adam(transformations, lr=hyperparameters.LEARNING_RATE)

# training loop
iterations = tqdm(range(hyperparameters.NUM_EPOCHS))

__deformedMeshFields__ = [] # to store the deformed mesh fields, global variable
__iteration__ = 0 # to keep track of the iterations, global variable

try:
    for __iteration__ in iterations:
        optimizer.zero_grad()

        __deformedMeshFields__ = []
        # transform the source objects
        for instance in range(1):
            src_meshField = __sourceMeshFields__[instance].clone()
            # upscaling the rotations and translations to force the params between -1 and 1
            R = (transformations[2*instance]) * 180.0
            T = transformations[2*instance+1]*environment.N
            src_meshField.transform_mesh(rotate=R, translate=T)
            src_meshField.transform_SDF(rotate=R, translate=T)
            __deformedMeshFields__.append(src_meshField)

        # setup the losses
        loss = {k: torch.tensor(0.0, device=device) for k in losses}

        # compute the losses
        for renderer, __targetTex__, __BnW__ in zip(renderers, target_silhouettes, target_bnw):
            loss["texture"] += Losses.get_silhouette_aware_texture_loss(renderer, __deformedMeshFields__, __BnW__, __targetTex__)
            # loss["LoG"] += Losses.get_LoG_loss_texture(renderer, __deformedMeshFields__, __targetTex__)

        loss["intersection"] = Losses.get_intersection_loss(environment, __deformedMeshFields__, hyperparameters.INTERSECTION_MODE)
        loss["bounding"] = Losses.get_environment_boundings_loss(__deformedMeshFields__, environment)

        # compute the net loss
        net_loss = torch.tensor(0.0, device=device)
        for k in losses:
            net_loss += losses[k]["weight"] * loss[k]
            losses[k]["values"].append(losses[k]["weight"] * loss[k].item())

        iterations.set_description("Texture: %.3f, Intersection: %.3f, Bounding: %.3f" % (losses["texture"]["values"][-1], losses["intersection"]["values"][-1], losses["bounding"]["values"][-1]))

        # backpropagate the loss
        net_loss.backward(retain_graph=True)
        optimizer.step()

        # visualize the results
        if hyperparameters.PROGRESSIVE_RESULTS:
            _silhouettes = []
            for i in range(hyperparameters.CAMERA_COUNT):
                renderer = renderers[i]
                silhouette = renderer.project_texture([mesh.mesh for mesh in __deformedMeshFields__])
                silhouette = silhouette.detach().cpu()
                _silhouettes.append(silhouette)
            visualizer.visualize_silhouette_multiple(_silhouettes, name=RUNS_FOLDER+'/silhouettes')
        
        if (__iteration__+1) % hyperparameters.PLOT_PERIOD == 0:
            _silhouettes = []
            for i in range(hyperparameters.CAMERA_COUNT):
                renderer = renderers[i]
                silhouette = renderer.project_texture([mesh.mesh for mesh in __deformedMeshFields__])
                silhouette = silhouette.detach().cpu()
                _silhouettes.append(silhouette)
                visualizer.visualize_silhouette_multiple(_silhouettes, name=RUNS_FOLDER + '/silhouettes/image' + '_' + str(__iteration__+1).zfill(6))

            visualizer.save_obj([i.mesh for i in __deformedMeshFields__], name=RUNS_FOLDER+'/report/structures/structure_' + str(__iteration__+1))

            if hyperparameters.VISUALIZE:
                sdf_points = [mesh.sdf_points.detach().cpu() for mesh in __deformedMeshFields__]
                sdf_values = [mesh.sdf_values.detach().cpu() for mesh in __deformedMeshFields__]
                visualizer.visualize_SDF(sdf_points, sdf_values, environment.environment, name=RUNS_FOLDER+'/report/SDF_' + str(__iteration__+1))
                environment.get_intersections(__deformedMeshFields__)
                visualizer.visualize_environment(environment, name=RUNS_FOLDER+'/report/intersection_' + str(__iteration__+1))

except KeyboardInterrupt:
    print("Training interrupted")

finally:
    # visualize the final results
    if not hyperparameters.VISUALIZE:
        sdf_points = [mesh.sdf_points.detach().cpu() for mesh in __deformedMeshFields__]
        sdf_values = [mesh.sdf_values.detach().cpu() for mesh in __deformedMeshFields__]
        visualizer.visualize_SDF(sdf_points, sdf_values, environment.environment, name=RUNS_FOLDER+'/report/SDF_' + str(__iteration__+1))
        environment.get_intersections(__deformedMeshFields__)
        visualizer.visualize_environment(environment, name=RUNS_FOLDER+'/report/intersection_' + str(__iteration__+1))
        
    # plot the losses
    visualizer.visualize_curves([
        {"values" : losses["texture"]["values"], "name" : "texture"},
        {"values" : losses["intersection"]["values"], "name" : "intersection"},
        {"values" : losses["bounding"]["values"], "name" : "bounding"},
        # {"values" : losses["LoG"]["values"], "name" : "LoG"}    
    ], name=RUNS_FOLDER+'/report/loss')

    # save the transformations in a text file
    save_file = os.path.join(RUNS_FOLDER, 'report', 'transformations.csv')
    header = 'Rotation_X, Rotation_Y, Rotation_Z, Translation_X, Translation_Y, Translation_Z\n'
    with open(save_file, 'w') as f:
        f.write(header)
        for t in range(0, len(transformations), 2):
            f.write(','.join([str(transformations[t][0].item()), str(transformations[t][1].item()), str(transformations[t][2].item()), 
                            str(transformations[t+1][0].item()), str(transformations[t+1][1].item()), str(transformations[t+1][2].item())]) + '\n')

    # save the losses in a csv file
    save_file = os.path.join(RUNS_FOLDER, 'report', 'losses.csv')
    header = 'Texture, Intersection, Bounding\n'
    with open(save_file, 'w') as f:
        f.write(header)
        for i in range(__iteration__):
            f.write(','.join([str(losses["texture"]["values"][i]), str(losses["intersection"]["values"][i]), str(losses["bounding"]["values"][i]), '\n']))
    
    visualizer.visualize_3d([i.mesh for i in __deformedMeshFields__], name=RUNS_FOLDER+'/report/structures/final')
    visualizer.save_obj([i.mesh for i in __deformedMeshFields__], name=RUNS_FOLDER+'/report/structures/final')
    visualizer.save_gif(RUNS_FOLDER+'/silhouettes/', RUNS_FOLDER+'/progressive.gif')
    visualizer.save_indivisual_objs([i.mesh for i in __deformedMeshFields__], name=RUNS_FOLDER+'/report/structures/individual')