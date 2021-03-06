from scipy.misc import imread
from pathlib import Path
import pandas as pd
import os

#data_dir = os.path.expanduser('~/datasets/rob535/')
data_dir = '/media/ryulion/Samsung_T5/ROB535_VISION_PROJECT'
print(data_dir)
#data_dir = os.path.join(Path(os.path.abspath(__file__)).parent, 'dataset')

assert os.path.exists(data_dir)
df = None
def init():
    global df
    label = Path(Path(os.path.abspath(__file__)).parent).parent
    print(label)

    df = pd.read_csv(os.path.join(label, 'labels.csv'))
def initialize(opt):
    return

def load_image(idx, is_train=True):
    image_row = df.loc[[idx]]
    path = (image_row['guid/image'].values[0])
    path = path + '_image.jpg'
    if is_train:
        parent_dir = os.path.join(data_dir, 'trainval')
        p = os.path.join(parent_dir, path)
    else:
        parent_dir = os.path.join(data_dir, 'test')
        p = os.path.join(parent_dir, path)
    return imread(p,mode='RGB')

def load_mask(idx, is_train=True):
	raise ValueError

def load_gt(idx):
    image_row = df.loc[[idx]]
    label = int(image_row['label'])
    return label

def setup_val_split(opt = None):
    #train = range(0, 19000)
    #train = range(10)
    #train = range(1500, 7573)
    train = range(7573)
    return train, train

def get_test_set():
    return range(1500)


if __name__ == "__main__":
    init()
    img = load_image(0)
