import os, sys, time, math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import qiskit
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


from Auxillary_functions import (
    MeasurementDataset,
    get_dict_from_array,
    Postprocessing, decorate_plot, normalize, AutoPlotter
)
from Model_learndist import QVAE_qcompile, create_qnn
from Training import QeVAE


#######################################################################
# SETUP PLOTTERS AND DATALOADERS
#######################################################################
 
def setup_dataloaders(want_datasetsize:float, train_size:float, n_samples):
    """Creates training and validation dataloaders. Print the size of each dataset"""

#     want_datasetsize=0.1; train_size = 0.7
    training_dataset = MeasurementDataset(datafile[: int(want_datasetsize*train_size * n_samples)])
    valid_dataset = MeasurementDataset(datafile[int(want_datasetsize*train_size * n_samples) :int(want_datasetsize*n_samples)])

    train_dataloader = DataLoader(
        training_dataset, batch_size=1, shuffle=True, num_workers=1
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=1, shuffle=True, num_workers=1
    )

    dataloaders = [train_dataloader, valid_dataloader]
    dataloader_info="Size of training set: %d | Size of validation set: %d"%(len(train_dataloader), len(valid_dataloader))
    print(dataloader_info)
    
    return dataloaders, dataloader_info
   
#######################################################################
# SETUP MAIN
#######################################################################

def main(learning_rates: list, dataloaders: list, true_results:dict, params: list):
    """Returns trained model, the path where results are written and the Earlystopper object"""

    nqubits, featuremap, patience, minibatchsize, beta, latentsize, num_epochs, annealing_schedule, nn_type, root_dir = params.values()
    training_params = {'patience':patience, 'minibatchsize':minibatchsize, 'beta':beta, 'annealing_schedule':annealing_schedule}
    qc_params = {'featuremap':featuremap, 'entanglement_type':'linear', 'repititions':1}
    
    qnn, qc = create_qnn(num_inputs=nqubits, num_qubits=nqubits, qc_params=qc_params)
    model = QVAE_qcompile(qnn, latentsize)

    encoder_lr, decoder_lr = learning_rates
    optimizer_encoder = torch.optim.Adam([i for i in list(model.parameters())[:-1]], lr=encoder_lr)
    optimizer_decoder = torch.optim.Adam([list(model.parameters())[-1]], lr=decoder_lr)

    # Print number of trainable parameters
    encoder_trainable_params = sum(
        p.numel() for p in model.encoder.parameters() if p.requires_grad
    )
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    trainparams_info = "No of trainable parameters: \n (Model:%d) | (Encoder:%d) | (Decoder:%d)"% (trainable_params, encoder_trainable_params, qnn.num_weights)
        
    epoch_batch_info = "Total number of epochs: %d | Total number of batches: %d"%(num_epochs,math.ceil(len(dataloaders[0])/minibatchsize))
    print(trainparams_info,'\n',epoch_batch_info)

    
    Earlystopping = QeVAE(
        training_params=training_params,
        dataloaders=dataloaders,
        optimizers=[optimizer_encoder, optimizer_decoder],
        true_result_keys=true_results.keys(),
        root_dir=root_dir,
        nn_type=nn_type
    )

    all_bitstrings = [("{0:0%db}"%nqubits).format(i) for i in range(pow(2, nqubits))]
    trained_model, directory_path = Earlystopping.start_training(model, all_bitstrings, num_epochs, true_results)

    return trained_model, directory_path, Earlystopping


#######################################################################
# RUN MAIN FROM TERMINAL
#######################################################################

if __name__ == "__main__":
    if len(sys.argv) == 13:
        # args sample: python3 main.py "nqubits_8_haar_seed43.npy" "Z" 7 0.001 0.009 64 1 0 40 fixed quantum-classical "./root_dir"
        print(sys.argv)
   
        # Get data filename
        filename = sys.argv[1] #"nqubits_8_haar_seed43.npy"
        
        datafile = np.load(filename)
        n_samples = datafile.shape[0]
        results = get_dict_from_array(datafile)
        true_results = get_dict_from_array(datafile)
        print("No of samples", n_samples)

        dataloaders, dataloader_info = setup_dataloaders(want_datasetsize=0.2, train_size=0.75, n_samples=n_samples)

        nqubits = datafile[0].shape[0]
        featuremap = sys.argv[2]
        patience = float(sys.argv[3])
        learning_rates = [float(sys.argv[4]), float(sys.argv[5])]
        minibatchsize = int(sys.argv[6])
        beta = float(sys.argv[7])
        latentsize = int(sys.argv[8])
        num_epochs = int(sys.argv[9])
        annealing_schedule = sys.argv[10]
        nn_type = sys.argv[11]
        root_dir = sys.argv[12]

        params = {'nqubits':nqubits, 'featuremap':featuremap, 'patience':patience,
                  'minibatchsize': minibatchsize, 'beta':beta, 'latentsize':latentsize,
                  'num_epochs':num_epochs, 'annealing_schedule':annealing_schedule, "nn_type" : nn_type,
                  'root_dir':root_dir }

        trained_model, directory_path, Earlystopper_obj = main(
            learning_rates,
            dataloaders,
            true_results,
            params
        )

        root_dir = os.path.dirname(directory_path)
        current_dir = directory_path.split('/')[-1]
        plotting_params = [nqubits, datafile]
        write_plots = AutoPlotter(root_dir, current_dir, Earlystopper_obj, plotting_params).write_plots()
        
    else:
        print("Error! \nWrong number of arguments. Please check")
    
     

