from torch.utils.data import Dataset
import pandas as pd
import torch


class SignatureDataset(Dataset):

    def __init__(self, dir, split='train'):
        self.file = self.extract(dir, split)

    @staticmethod
    def extract(dir, split):
        """

        :param data:
        :param dir: Directory of images
        :return:
        """
        try:
            data = pd.read_csv(dir + f'{split}_data.csv', names=['sample_a', 'sample_b', 'target'])

        except:
            raise FileNotFoundError(f'Could not read file {dir}{split}_data.csv')

        data[['a_id', 'a_img']] = data['sample_a'].str.split('/').to_list()
        data[['b_id', 'b_img']] = data['sample_b'].str.split('/').to_list()
        data['forged'] = data['b_id'].str.contains('forg')

        # Adding
        data['sample_a'] = dir + split + '/' + data['sample_a']
        data['sample_b'] = dir + split + '/' + data['sample_b']
        return data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
