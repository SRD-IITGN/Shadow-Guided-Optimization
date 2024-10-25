# environment parameters
ENVIRONMENT_SIZE = 20000

# camera parameters
CAMERA_TRANSFORMS = [
    [None,   0.0,   0.0],   # Front view
    [None,   0.0,  90.0],   # Right side view
    [None,   0.0, 180.0],   # Back view
    [None,   0.0, 270.0],   # Left side view
    [None,  90.0,   0.0],   # Top view
    [None, -90.0,   0.0],   # Bottom view
]
CAMERA_TYPES = ['perspective']*len(CAMERA_TRANSFORMS)  # 'orthographic' or 'perspective'
FACES_PER_PIXEL = 100
IMAGE_SIZE = 150

# object parameters
TARGET_OBJECTS = ['input/piece_0.obj', 
                  'input/piece_1.obj', 
                  'input/piece_2.obj'] 
TARGET_SCALE = 0.8
SOURCE_OBJECTS = TARGET_OBJECTS
SOURCE_SCALE = TARGET_SCALE
RANDOM_INITIAL_TRANSFORMATIONS = False

# training parameters
NUM_EPOCHS = 5000
WEIGHT_SILHOUETTE = 10.0
WEIGHT_INTERSECTION = 0.1
WEIGHT_DISTANCE = 0.000
WEIGHT_BOUNDING = 1
WEIGHT_LOG = 1.0
LEARNING_RATE = 1e-1
INTERSECTION_MODE = 'counting'

# visualization parameters
PLOT_PERIOD = NUM_EPOCHS // 100
PROGRESSIVE_RESULTS = True
VISUALIZE = False

# derived hyperparameters
CAMERA_COUNT = len(CAMERA_TRANSFORMS)
NUM_OBJECTS = len(SOURCE_OBJECTS)
