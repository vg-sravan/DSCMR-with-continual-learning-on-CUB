import torch
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

from datetime import datetime
import torch.optim as optim
import pickle
import numpy as np
import matplotlib.pyplot as plt
from model import IDCM_NN
from train_model import train_model
from load_data import get_loader
from load_data import get_datasets
from load_data import get_params
from train_model import testing_model
from evaluate import fx_calc_map_label
import os
######################################################################
# Start running

if __name__ == '__main__':
    # environmental setting: setting the following parameters based on your experimental environment.
    dataset = 'pascal'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # data parameters
    DATA_DIR = 'data/' + dataset + '/'
    alpha = 1e-3
    beta = 1e-1
    MAX_EPOCH = 15
    batch_size = 60
    testing_batch_size = batch_size
    # batch_size = 512
    lr = 1e-4
    betas = (0.5, 0.999)
    weight_decay = 0
    num_tasks = 20

    print('Datasets creating.........')

    train_dict, test_dict, size_dict = get_datasets(DATA_DIR)

    print('Datasets created.........')

    print('Getting parameters............')

    # data_loader, input_data_par = get_loader(DATA_DIR, batch_size)

    input_data_par = get_params(train_dict['1'], test_dict['1'])

    print('Got parameters................')

    print('Creating Model....... and setting up')

    model_ft = IDCM_NN(img_input_dim=input_data_par['img_dim'],
                       text_input_dim=input_data_par['text_dim'], output_dim=input_data_par['num_class']).to(device)
    # params_to_update = list(model_ft.parameters())
    params_to_update = list(model_ft.img_net.parameters())
    params_to_update += (list(model_ft.text_net.parameters()))
    params_to_update += (list(model_ft.linearLayer.parameters()))
    params_to_update += (list(model_ft.linearLayer2.parameters()))

    # Observe that all parameters are being optimized
    optimizer = optim.Adam(params_to_update, lr=lr, betas=betas)

    hist_dict = {'img_mAP': {}, 'txt_mAP': {}, 'loss': {}}
    mAP_dict = {str(i):[] for i in range(1, num_tasks+1)}
    print('Created Model.......and setup')
    print('..........................Training is beginning..............................')

    for curr_dataset_, _ in train_dict.items():

        print(f'Selected dataset for training is {curr_dataset_}..........')

        curr_dataset_train = train_dict[curr_dataset_]
        curr_dataset_test = test_dict[curr_dataset_]
        curr_dataset_number = int(curr_dataset_)

        curr_dataloader, curr_params = get_loader(
            curr_dataset_train, curr_dataset_test, batch_size)

        model_ft, img_acc_hist, txt_acc_hist, loss_hist = train_model(
            model_ft, curr_dataloader, optimizer, alpha, beta, curr_dataset_, num_epochs=MAX_EPOCH)

        hist_dict['img_mAP'][curr_dataset_] = img_acc_hist
        hist_dict['txt_mAP'][curr_dataset_] = txt_acc_hist
        hist_dict['loss'][curr_dataset_] = loss_hist

        print(f'Starting testing on datasets trained till now..............')
        for testing_idx in range(1, curr_dataset_number+1):

            testing_name = str(testing_idx)
            testing_dataset_train = train_dict[testing_name]
            testing_dataset_test = test_dict[testing_name]

            testing_dataloader, testing_params = get_loader(
                testing_dataset_train, testing_dataset_test, testing_batch_size)

            mAP_dict[testing_name].append(testing_model(model_ft, testing_dataloader, testing_name))

    print('........................Training is completed...............................')

    if not os.path.isdir('../Results'):
        os.mkdir('../Results')
        
    results_outpath = '../Results/'

    mAP_outfile = results_outpath + 'mAPs.pickle'
    with open(mAP_outfile, 'wb') as f_out:
        pickle.dump(mAP_dict, f_out)
        print('mAP history saved to: ', mAP_outfile)
    
    print(f'Plotting.............................')
    
    if not os.path.isdir('../Results/plots'):
        os.mkdir('../Results/plots')
    
    for plt_dataset, plt_vals in mAP_dict.items():
        
        plt.title(f'mAP for testing data of dataset{plt_dataset} vs timestamp')
        plt.plot(range(int(plt_dataset), num_tasks+1), plt_vals)
        plt.xlabel(f"dataset timestamp")
        plt.ylabel(f"Average mean Average Precision")
        plt.xticks(np.arange(1, num_tasks+1, 1.0))
        plt.savefig(f"../Results/plots/mAP_{plt_dataset}.png")
        plt.close()
         
    print(f'Done plotting....................')
    
    print('.....well!!!! done')

#     print('...Evaluation on testing data...')
#     view1_feature, view2_feature, view1_predict, view2_predict = model_ft(torch.tensor(
#         input_data_par['img_test']).to(device), torch.tensor(input_data_par['text_test']).to(device))
#     label = torch.argmax(torch.tensor(input_data_par['label_test']), dim=1)
#     view1_feature = view1_feature.detach().cpu().numpy()
#     view2_feature = view2_feature.detach().cpu().numpy()
#     view1_predict = view1_predict.detach().cpu().numpy()
#     view2_predict = view2_predict.detach().cpu().numpy()
#     img_to_txt = fx_calc_map_label(view1_feature, view2_feature, label)
#     print('...Image to Text MAP = {}'.format(img_to_txt))

#     txt_to_img = fx_calc_map_label(view2_feature, view1_feature, label)
#     print('...Text to Image MAP = {}'.format(txt_to_img))

#     print('...Average MAP = {}'.format(((img_to_txt + txt_to_img) / 2.)))
