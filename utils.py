from torch.utils.data import Dataset
from skimage import io, transform
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from torchvision import transforms


class SignatureDataset(Dataset):
    def __init__(self, root, split='train', transforms=None):
        self.file = self.extract(root, split)
        self.transforms = transforms

    @staticmethod
    def extract(root, split):
        """

        :param data:
        :param root: Directory of images
        :return:
        """
        try:
            data = pd.read_csv(root + f'{split}_data.csv',
                               names=['sample_a', 'sample_b', 'target'])

        except:
            raise FileNotFoundError(
                f'Could not read file {root}{split}_data.csv')

        data[['a_id', 'a_img']] = data['sample_a'].str.split('/').to_list()
        data[['b_id', 'b_img']] = data['sample_b'].str.split('/').to_list()
        data['forged'] = data['b_id'].str.contains('forg')

        # Adding
        data['sample_a'] = root + split + '/' + data['sample_a']
        data['sample_b'] = root + split + '/' + data['sample_b']
        return data

    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_a_name = self.file.loc[idx, 'sample_a']
        img_b_name = self.file.loc[idx, 'sample_b']
        is_match = self.file.loc[idx, 'target']
        img_a = io.imread(img_a_name)
        img_b = io.imread(img_b_name)
        sample = {'img_a': img_a, 'img_b': img_b, 'is_match': is_match}

        if self.transforms:
            sample = self.transforms(sample)

        return sample


class Rescale:
    def __init__(self, size):
        assert isinstance(size, (int, tuple))
        if isinstance(size, int):
            self.size = (size, size)
        else:
            assert len(size) == 2
            self.size = size

    def __call__(self, sample):
        img_a, img_b, is_match = sample['img_a'], sample['img_b'], sample[
            'is_match']
        h, w = self.size
        img_a = transform.resize(img_a, (h, w))
        img_b = transform.resize(img_b, (h, w))

        return {'img_a': img_a, 'img_b': img_b, 'is_match': is_match}


class ToTensor:
    """Convert ndarrays to torch tensor"""

    def __call__(self, sample):
        img_a, img_b, is_match = sample['img_a'], sample['img_b'], sample['is_match']

        # Bring color channel to first position
        img_a = img_a.transpose((2, 0, 1))
        img_b = img_b.transpose((2, 0, 1))

        return {'img_a': torch.from_numpy(img_a),
                'img_b': torch.from_numpy(img_b),
                'is_match': torch.from_numpy(np.array([is_match]))}



if __name__ == '__main__':
    rescale = Rescale((100, 100))
    train = SignatureDataset(root='./data/sign_data/',
                             split='train',
                             transforms=transforms.Compose([rescale,
                                                            ToTensor()]))

