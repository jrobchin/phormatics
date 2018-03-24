# Paths
BASE_DIR = os.path.abspath('..')
GRAPH_PATH = os.path.join(BASE_DIR, 'models', 'mobilenet', 'graph_opt.pb')
DATA_PATHS = {'base': os.path.join(BASE_DIR, 'data'),
              'sample': os.path.join(BASE_DIR, 'data', 'sample')}

# Processing
PROCESSING_DIMS = (240, 240)

# Camera
CAMERA_NUMBER = 1