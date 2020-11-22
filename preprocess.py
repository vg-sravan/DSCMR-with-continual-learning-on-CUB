import numpy as np
import pickle
import imageio
from skimage.transform import resize
import os


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


IMG_SZE = 256
picklepath = '../Data/changed_data'
embedding_filename = '/embeddings.pickle'

with open(picklepath + '/filenames.pickle', 'rb') as f:
    list_filenames1 = pickle.load(f)
    list_filenames1 = np.array(list_filenames1)
with open(picklepath + '/classes.pickle', 'rb') as f1:
    labels_1 = pickle.load(f1, encoding="bytes")
    labels_1 = np.array(labels_1)
with open(picklepath + embedding_filename, 'rb') as f:
    embeddings_1 = pickle.load(f, encoding="bytes")
    embeddings_1 = np.array(embeddings_1)
# image preprocessing
list_filenames = []
labels_ = []
embeddings_ = []
for idx in range(len(list_filenames1)):
    i = list_filenames1[idx]
    f_name = f'../Data/images/CUB_200_2011/images/{i}.jpg'
    _img = get_image(f_name, IMG_SZE, is_crop=True)
    _img = _img.astype('uint8')
    _imag = np.array(_img)
    if(len(_imag.shape) == 3):
        list_filenames.append(list_filenames1[idx])
        labels_.append(labels_1[idx])
        embeddings_.append(embeddings_1[idx])
list_filenames = np.array(list_filenames)
labels_ = np.array(labels_)
embeddings_ = np.array(embeddings_)

outpath = '../Data/changed_data/preprocessed_data/'

embed_outfile = outpath + 'embeddings.pickle'
with open(embed_outfile, 'wb') as f_out:
    pickle.dump(embeddings_, f_out)
    print('Preprocessed Embeddings saved to: ', embed_outfile)

filenames_outfile = outpath + 'filenames.pickle'
with open(filenames_outfile, 'wb') as f_out:
    pickle.dump(list_filenames, f_out)
    print('Preprocessed Filenames saved to: ', filenames_outfile)

classes_outfile = outpath + 'labels.pickle'
with open(classes_outfile, 'wb') as f_out:
    pickle.dump(labels_, f_out)
    print('Preprocessed Labels saved to: ', classes_outfile)

    print("Image PP Done")
