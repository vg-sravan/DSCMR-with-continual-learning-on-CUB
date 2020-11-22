import numpy as np
import pickle


with open('../Data/texts/train/char-CNN-RNN-embeddings.pickle', 'rb') as f:
    embeddings_train = pickle.load(f, encoding="bytes")
    embeddings_train = np.array(embeddings_train)

with open('../Data/texts/test/char-CNN-RNN-embeddings.pickle', 'rb') as f:
    embeddings_test = pickle.load(f, encoding="bytes")
    embeddings_test = np.array(embeddings_test)

embeddings_total = np.concatenate((embeddings_train, embeddings_test), axis=0)

with open('../Data/texts/train/filenames.pickle', 'rb') as f:
    filenames_train = pickle.load(f, encoding="bytes")
    filenames_train = np.array(filenames_train)


with open('../Data/texts/test/filenames.pickle', 'rb') as f:
    filenames_test = pickle.load(f, encoding="bytes")
    filenames_test = np.array(filenames_test)

filenames_total = np.concatenate((filenames_train, filenames_test), axis=0)


with open('../Data/texts/train/class_info.pickle', 'rb') as f:
    classes_train = pickle.load(f, encoding="bytes")
    classes_train = np.array(classes_train)


with open('../Data/texts/test/class_info.pickle', 'rb') as f:
    classes_test = pickle.load(f, encoding="bytes")
    classes_test = np.array(classes_test)

classes_total = np.concatenate((classes_train, classes_test), axis=0)

outpath = '../Data/changed_data/'

embed_outfile = outpath + 'embeddings.pickle'
with open(embed_outfile, 'wb') as f_out:
    pickle.dump(embeddings_total, f_out)
    print('Embeddings saved to: ', embed_outfile)

filenames_outfile = outpath + 'filenames.pickle'
with open(filenames_outfile, 'wb') as f_out:
    pickle.dump(filenames_total, f_out)
    print('Filenames saved to: ', filenames_outfile)

classes_outfile = outpath + 'classes.pickle'
with open(classes_outfile, 'wb') as f_out:
    pickle.dump(classes_total, f_out)
    print('Classes saved to: ', classes_outfile)
