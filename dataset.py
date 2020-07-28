import cv2
import numpy as np

data_root = './data'

def load_MNIST(size=None):
    # load MNIST dataset
    # from [0,1] float32
    from pathlib import Path
    import requests
    DATA_PATH = Path(data_root)
    PATH = DATA_PATH / "mnist"
    PATH.mkdir(parents=True, exist_ok=True)
    URL = "http://deeplearning.net/data/mnist/"
    FILENAME = "mnist.pkl.gz"
    if not (PATH / FILENAME).exists():
            content = requests.get(URL + FILENAME).content
            (PATH / FILENAME).open("wb").write(content)
    import pickle
    import gzip
    with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
            ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
    x_train = x_train.reshape((50000,28,28))
    x_valid = x_valid.reshape((10000,28,28))
    if size:
        x_train = np.array([cv2.resize(x,(size,size)) for x in x_train])
        x_valid = np.array([cv2.resize(x,(size,size)) for x in x_valid])
    print(f"Loaded MNIST dataset: x_train{x_train.shape}, x_valid{x_valid.shape}")
    return x_train, x_valid


def load_EMNIST(size=None):
    import numpy as np
    from emnist import extract_training_samples
    images, labels = extract_training_samples('letters')
    x_train = (images[:100000]/255).astype(np.float32)
    x_valid = (images[100000:]/255).astype(np.float32)
    if size:
        x_train = np.array([cv2.resize(x,(size,size)) for x in x_train])
        x_valid = np.array([cv2.resize(x,(size,size)) for x in x_valid])
    print(f"Loaded EMNIST dataset: x_train{x_train.shape}, x_valid{x_valid.shape}")
    return x_train, x_valid


def load_CIFAR10(size=None, channel = -1):
    import torchvision
    import numpy as np
    root = data_root
    CIFAR10_train = torchvision.datasets.CIFAR10(root, train=True, download=True)
    CIFAR10_test = torchvision.datasets.CIFAR10(root, train=False, download=True)


    if channel == -1:
        rgb2gray = np.array([[0.2989], [0.5870], [0.1140]])
        x_train = np.squeeze(CIFAR10_train.data@rgb2gray)
        x_valid = np.squeeze(CIFAR10_test.data@rgb2gray)
        print('using Gray image')
    else:
        print(f'using channel {channel}')
        x_train = CIFAR10_train.data[:,:,:,channel]
        x_valid = CIFAR10_test.data[:,:,:,channel]

    x_train = (x_train/255).astype(np.float32)
    x_valid = (x_valid/255).astype(np.float32)
    if size:
        x_train = np.array([cv2.resize(x,(size,size)) for x in x_train])
        x_valid = np.array([cv2.resize(x,(size,size)) for x in x_valid])
    print(f"Loaded CIFAR10 gray dataset: x_train{x_train.shape}, x_valid{x_valid.shape}")
    return x_train, x_valid


def load_FMNIST(size=None):
    import torchvision
    import numpy as np
    root = data_root
    FMNIST_train = torchvision.datasets.FashionMNIST(root, train=True, download=True)
    FMNIST_test = torchvision.datasets.FashionMNIST(root, train=False, download=True)
    x_train = (FMNIST_train.data.numpy()/255).astype(np.float32)
    x_valid = (FMNIST_test.data.numpy()/255).astype(np.float32)
    if size:
        x_train = np.array([cv2.resize(x,(size,size)) for x in x_train])
        x_valid = np.array([cv2.resize(x,(size,size)) for x in x_valid])
    print(f"Loaded FashionMNIST gray dataset: x_train{x_train.shape}, x_valid{x_valid.shape}")
    return x_train, x_valid


def load_SVHN(size=None, channel=-1):
    import torchvision
    import numpy as np
    root = data_root
    SVHN_train = torchvision.datasets.SVHN(root, split='train', download=True)
    SVHN_test = torchvision.datasets.SVHN(root, split='test', download=True)
    x_train = np.moveaxis(SVHN_train.data, 1, -1)
    x_valid = np.moveaxis(SVHN_test.data, 1, -1)

    if channel == -1:
        rgb2gray = np.array([[0.2989], [0.5870], [0.1140]])
        x_train = np.squeeze(x_train@rgb2gray)
        x_valid = np.squeeze(x_valid@rgb2gray)
        print('using Gray image')
    elif channel in range(3):
        print(f'using channel {channel}')
        x_train = x_train[:,:,:,channel]
        x_valid = x_test[:,:,:,channel]

    x_train = (x_train/255).astype(np.float32)
    x_valid = (x_valid/255).astype(np.float32)
    if size:
        x_train = np.array([cv2.resize(x,(size,size)) for x in x_train])
        x_valid = np.array([cv2.resize(x,(size,size)) for x in x_valid])
    print(f"Loaded SVHN gray dataset: x_train{x_train.shape}, x_valid{x_valid.shape}")
    return x_train, x_valid


def load_CELEBA(size = None, N_1k = 0, channel = -1):
    from pathlib import Path
    import matplotlib.image as mpimg
    import numpy as np

    data_dir = f"{data_root}/img_align_celeba"
    data_path = Path(data_dir)
    names = sorted(list(data_path.glob("*.jpg")))
    N = len(names)
    m,n,c = mpimg.imread(names[0]).shape
    # len = 202599
    N_load = 1200
    images_rgb = np.zeros([N_load,m,n,c])
    for i in range(N_load):
        images_rgb[i] = mpimg.imread(names[N_1k*N_load+i])

    if channel == -1:
        rgb2gray = np.array([[0.2989], [0.5870], [0.1140]])
        images_gray = np.squeeze(images_rgb@rgb2gray)
        print('using Gray image')
        img_out = images_gray
    else:
        print(f'using channel {channel}')
        img_out = images_rgb[:,:,:,channel]

    x_train = img_out[:200]
    x_valid = img_out[200:]

    x_train = (x_train/255).astype(np.float32)
    x_valid = (x_valid/255).astype(np.float32)
    x_train = x_train[:,109-89:109+89,:]
    x_valid = x_valid[:,109-89:109+89,:]
    if size:
        x_train = np.array([cv2.resize(x,(size,size)) for x in x_train])
        x_valid = np.array([cv2.resize(x,(size,size)) for x in x_valid])
    print(f"Loaded CELEBA gray dataset: x_train{x_train.shape}, x_valid{x_valid.shape}")
    return x_train, x_valid
