from torch.utils.data.dataset import Dataset
import torch.utils.data as data
from scipy.io import loadmat, savemat
from torch.utils.data import DataLoader
import torch
import numpy as np
import imageio
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import pickle
import scipy.misc
import errno
import os

IMG_SZE = 256


def transform(image, image_size, is_crop):
    transformed_image = resize(image, [image_size, image_size], order=3)
    return transformed_image


def imread(path):
    img = imageio.imread(path)
    if len(img.shape) == 0:
        raise ValueError(path + " got loaded as a dimensionless array!")
    return img.astype(np.float)


def get_image(image_path, image_size, is_crop=False, bbox=None):
    # global index
    out = transform(imread(image_path), image_size, is_crop)
    return out


def ind2vec(ind, N=None):
    ind = np.asarray(ind)
    if N is None:
        N = ind.max() + 1
    if(len(ind.shape) == 2):
        return np.arange(N) == np.repeat(ind, N, axis=1)
    else:
        return np.arange(N) == np.repeat(ind, N)


class CustomDataSet(data.Dataset):
    def __init__(
            self,
            image_files,
            texts,
            labels):
        self.image_files = image_files
        self.texts = texts
        self.labels = labels
        self.number_classes = len(np.unique(labels))

    def __getitem__(self, index):
        img_file = self.image_files[index]
        f_name = f'../Data/images/CUB_200_2011/images/{img_file}.jpg'
        img = get_image(f_name, IMG_SZE, is_crop=True)
        img = img.astype('uint8')
        _image = np.array(img)
        image = torch.from_numpy(_image)
#         print('In here\n')
#         print(f'{image.shape}, path = {f_name}')
        image = image.permute(2, 0, 1)
        image = image.float()
        text = self.texts[index]
        label = self.labels[index]
        return image, text, label

    def __len__(self):
        count = len(self.image_files)
        assert len(self.image_files) == len(self.labels)
        return count


class AppendName(data.Dataset):
    """
    A dataset wrapper that also return the name of the dataset/task
    """

    def __init__(self, dataset, name, first_class_ind=0):
        super(AppendName, self).__init__()
        self.dataset = dataset
        self.name = name
        self.first_class_ind = first_class_ind  # For remapping the class index

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, txt, label = self.dataset[index]
        label = label + self.first_class_ind
        return img, txt, label, self.name


class Subclass(data.Dataset):
    """
    A dataset wrapper that return the task name and remove the offset of labels (Let the labels start from 0)
    """

    def __init__(self, dataset, class_list, remap=True):
        '''
        :param dataset: (CacheClassLabel)
        :param class_list: (list) A list of integers
        :param remap: (bool) Ex: remap class [2,4,6 ...] to [0,1,2 ...]
        '''
        super(Subclass, self).__init__()
        assert isinstance(
            dataset, CustomDataSet), 'dataset must be wrapped by CacheClassLabel'
        self.dataset = dataset
        self.class_list = class_list
        self.remap = remap
        self.indices = []
        for c in class_list:
            to_extend = (dataset.labels == c).nonzero().flatten().tolist() if isinstance(
                dataset.labels, torch.Tensor) else list((dataset.labels == c).nonzero()[0])
            self.indices.extend(to_extend)
        if remap:
            self.class_mapping = {c: i for i, c in enumerate(class_list)}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        img, txt, label = self.dataset[self.indices[index]]
        if self.remap:
            raw_label = label.item() if isinstance(label, torch.Tensor) else label
            label = self.class_mapping[raw_label]
        label = (ind2vec(label, len(self.class_list))).astype(int)
        return img, txt, label


def SplitGen(train_dataset, val_dataset, first_split_sz=10, other_split_sz=10, rand_split=False, remap_class=True):
    '''
    Generate the dataset splits based on the labels.
    :param train_dataset: (torch.utils.data.dataset)
    :param val_dataset: (torch.utils.data.dataset)
    :param first_split_sz: (int)
    :param other_split_sz: (int)
    :param rand_split: (bool) Randomize the set of label in each split
    :param remap_class: (bool) Ex: remap classes in a split from [2,4,6 ...] to [0,1,2 ...]
    :return: train_loaders {task_name:loader}, val_loaders {task_name:loader}, out_dim {task_name:num_classes}
    '''
    assert train_dataset.number_classes == val_dataset.number_classes, 'Train/Val has different number of classes'
    num_classes = train_dataset.number_classes

    # Calculate the boundary index of classes for splits
    # Ex: [0,2,4,6,8,10] or [0,50,60,70,80,90,100]
    split_boundaries = [0, first_split_sz]
    while split_boundaries[-1] < num_classes:
        split_boundaries.append(split_boundaries[-1]+other_split_sz)
    print('split_boundaries:', split_boundaries)
    assert split_boundaries[-1] == num_classes, 'Invalid split size'

    # Assign classes to each splits
    # Create the dict: {split_name1:[2,6,7], split_name2:[0,3,9], ...}
    if not rand_split:
        class_lists = {str(i): list(range(
            split_boundaries[i-1]+1, split_boundaries[i]+1)) for i in range(1, len(split_boundaries))}
    else:
        randseq = torch.randperm(num_classes)
        class_lists = {str(i): randseq[list(range(
            split_boundaries[i-1], split_boundaries[i]))].tolist() for i in range(1, len(split_boundaries))}
    print(class_lists)

    # Generate the dicts of splits
    # Ex: {split_name1:dataset_split1, split_name2:dataset_split2, ...}
    train_dataset_splits = {}
    val_dataset_splits = {}
    task_output_space = {}
    for name, class_list in class_lists.items():
        train_dataset_splits[name] = AppendName(
            Subclass(train_dataset, class_list, remap_class), name)
        val_dataset_splits[name] = AppendName(
            Subclass(val_dataset, class_list, remap_class), name)
        task_output_space[name] = len(class_list)

    return train_dataset_splits, val_dataset_splits, task_output_space


def get_datasets(path, use_path=False):
    if use_path:
        picklepath = path
    else:
        picklepath = '../Data/changed_data/preprocessed_data'

    with open(picklepath + '/filenames.pickle', 'rb') as f:
        list_filenames = pickle.load(f)
        list_filenames = np.array(list_filenames)
    with open(picklepath + '/labels.pickle', 'rb') as f1:
        labels = pickle.load(f1, encoding="bytes")
        labels = np.array(labels)
    with open(picklepath + '/embeddings.pickle', 'rb') as f:
        embeddings = pickle.load(f, encoding="bytes")
        embeddings = np.array(embeddings)

    train_filenames, test_filenames, train_embeddings, test_embeddings, train_labels, test_labels = train_test_split(
        list_filenames, embeddings, labels, test_size=0.2, random_state=42, stratify=labels)

    train_embeddings = np.mean(train_embeddings, axis=1, keepdims=True)
    test_embeddings = np.mean(test_embeddings, axis=1, keepdims=True)
    # train_embeddings = train_embeddings[:, 0, :]
    # test_embeddings = test_embeddings[:, 0, :]
    train_embeddings = train_embeddings.squeeze()
    test_embeddings = test_embeddings.squeeze()

    # label_train = label_train[:, None]
    # label_test = label_test[:, None]

    # label_train = ind2vec(label_train).astype(int)
    # label_test = ind2vec(label_test).astype(int)

    train_dataset = CustomDataSet(
        train_filenames, train_embeddings, train_labels)
    test_dataset = CustomDataSet(test_filenames, test_embeddings, test_labels)

    train_dict, test_dict, size_dict = SplitGen(
        train_dataset, test_dataset, 10, 10)

    return train_dict, test_dict, size_dict


def get_loader(train_dataset, test_dataset, batch_size):
    # img_train = loadmat(path+"train_img.mat")['train_img']
    # img_test = loadmat(path + "test_img.mat")['test_img']
    # text_train = loadmat(path+"train_txt.mat")['train_txt']
    # text_test = loadmat(path + "test_txt.mat")['test_txt']
    # label_train = loadmat(path+"train_img_lab.mat")['train_img_lab']
    # label_test = loadmat(path + "test_img_lab.mat")['test_img_lab']

    # img_train = loadmat(path+"train_img.mat")['train_img']
    # img_test = loadmat(path + "test_img.mat")['test_img']
    # text_train = loadmat(path+"train_txt.mat")['train_txt']
    # text_test = loadmat(path + "test_txt.mat")['test_txt']
    # label_train = loadmat(path+"train_img_lab.mat")['train_img_lab']
    # label_test = loadmat(path + "test_img_lab.mat")['test_img_lab']

    # imgs = {'train': train_filenames, 'test': test_filenames}
    # texts = {'train': train_embeddings, 'test': test_embeddings}
    # labels = {'train': label_train, 'test': label_test}
    # dataset = {x: CustomDataSet(image_files=imgs[x], texts=texts[x], labels=labels[x])
    #            for x in ['train', 'test']}

    img, embed, label, namee = train_dataset[0]

    print(f'Starting preparing dataloader for dataset {namee}............')
    dataset = {'train': train_dataset, 'test': test_dataset}

    shuffle = {'train': False, 'test': False}

    dataloader = {x: DataLoader(dataset[x], batch_size=batch_size,
                                shuffle=shuffle[x], num_workers=0) for x in ['train', 'test']}

    img_dim = 4096  # from vggnet
    text_dim = embed.shape[-1]
    num_class = label.shape[-1]

    input_data_par = {}
    # input_data_par['img_test'] = imgs['test']
    # input_data_par['text_test'] = texts['test']
    # input_data_par['label_test'] = labels['test']
    # input_data_par['img_train'] = imgs['train']
    # input_data_par['text_train'] = texts['train']
    # input_data_par['label_train'] = labels['train']
    input_data_par['img_dim'] = img_dim
    input_data_par['text_dim'] = text_dim
    input_data_par['num_class'] = num_class
    input_data_par['name'] = namee

    print(f'Done preparing dataloader for dataset {namee}')
    return dataloader, input_data_par
    # return dataloader


def get_params(train_dataset, test_dataset,):
    # ''' Getting parameters from the dataset '''
    img, embed, label, namee = train_dataset[0]

    img_dim = 4096  # from vggnet
    text_dim = embed.shape[-1]
    num_class = label.shape[-1]

    input_data_par = {}

    input_data_par['img_dim'] = img_dim
    input_data_par['text_dim'] = text_dim
    input_data_par['num_class'] = num_class
    input_data_par['name'] = namee

    return input_data_par
