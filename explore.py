import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl






"""
STRUCTURE 
.
├── test
│		 ├── 049
│		 │		 ├── 01_049.png
│		 │		 ├── 02_049.png
│		 │		 ├── 03_049.png
│		 │		 ├── 04_049.png
│		 │		 ├── 05_049.png
│		 │		 ├── 06_049.png
│		 │		 ├── 07_049.png
│		 │		 ├── 08_049.png
│		 │		 ├── 09_049.png
│		 │		 ├── 10_049.png
│		 │		 ├── 11_049.png
│		 │		 └── 12_049.png
├── test_data.csv
├── train
└── train_data.csv


"""


DATADIR = './data/sign_data/'
train_file = pd.read_csv(DATADIR + 'train_data.csv', names=['sample_a', 'sample_b', 'target'])
test_file = pd.read_csv(DATADIR + 'test_data.csv', names=['sample_a', 'sample_b', 'target'])

def extract(data):
    data[['a_id', 'a_img']] = data['sample_a'].str.split('/').to_list()
    data[['b_id', 'b_img']] = data['sample_b'].str.split('/').to_list()
    data['forged'] = data['b_id'].str.contains('forg')
    return data

train_file = extract(train_file)
test_file = extract(test_file)



def show_pic(path, train=True):
    if train:
        path = DATADIR + 'train/' + path

    else:
        path = DATADIR + 'train/' + path

    img = plt.imread(path)
    plt.imshow(img)
    plt.show()

def compare_pic(number):
    folder = DATADIR + 'train/'

    img_a_path = folder + train_file.iloc[number]['sample_a']

    img_b_path = folder + train_file.iloc[number]['sample_b']
    matching = train_file.iloc[number]['target']
    forgery = train_file.iloc[number]['forged']

    img_a = plt.imread(img_a_path)
    img_b = plt.imread(img_b_path)
    a_id = train_file.iloc[number]['a_id']
    b_id = train_file.iloc[number]['b_id']



    fig = plt.figure()

    title = ''
    if matching:
        title += 'Signatures: Match\n'

    else:
        title += 'Signatures: No Macht\n'

    if forgery:
        title += 'Signature: Forged'

    else:
        title += 'Signature: Authentic'

    plt.title(f'{title}')

    ax = fig.add_subplot(1, 2, 1)
    img = plt.imshow(img_a)
    plt.title(f'ID: {train_file.iloc[number]["a_id"]}\n'
              f'IMG: {train_file.iloc[number]["a_img"]}')
    ax = fig.add_subplot(1, 2, 2)
    img = plt.imshow(img_b)
    plt.title(f'ID: {train_file.iloc[number]["b_id"]}\n'
          f'IMG: {train_file.iloc[number]["b_img"]}')


    plt.show()

# TODO: Add Forgery description to plot
# TODO: Find out how many pictures are matched
# TODO: Find out how many signatures are from same person but target is 0
# TODO: Person identifier in compare pic wrong


subset = train_file.sample(5).copy()
sample_row = subset.values[0]


def forgery_checker(row):
    # Check spellings
    # Check same
    if row[0] == row[1]:
        return 1

    else:
        return 0

checked = train_file.apply(forgery_checker, axis=1)

