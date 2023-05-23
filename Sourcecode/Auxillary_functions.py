import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset

now = lambda x: "_".join(datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p").split(":"))

def decorate_plot(ax):
    sns.set_theme(style="ticks")
    ax.minorticks_on()
    ax.tick_params(direction="in", right=True, top=True)
    ax.tick_params(labelsize=18)
    ax.tick_params(labelbottom=True, labeltop=False, labelright=False)
    ax.tick_params(direction="in", which="minor", length=5, bottom=True)
    ax.tick_params(direction="in", which="major", length=10, bottom=True)
    ax.grid()
    return ax
    
def normalize(d, target=1.0):
   raw = sum(d.values())
   factor = target/raw
   return {key:value*factor for key,value in d.items()}

def conv_listbitstrings_numpy(strings_array:np.ndarray, nqubits) -> np.ndarray:
    """Converts a 1D numpy array of bit-strings into a 2D numpy array
    where each row contains the bits data"""
    
    output_array = np.zeros((strings_array.shape[0], nqubits))
    for rowid, row in enumerate(strings_array):
        for colid, elem in enumerate(row):
            output_array[rowid, colid] = elem
    return output_array

def get_dict_from_array(datafile:np.ndarray):
    counter = 0
    results_dict = {}
    for row in datafile:
        as_str = row.astype(int).astype(str)
        output = str(np.apply_along_axis(''.join, 0, as_str))

        if output in results_dict:
            results_dict[output] += 1
        else:
            results_dict[output] = 1
    return results_dict

class MeasurementDataset(Dataset):

    def __init__(self, numpy_array:np.ndarray):
        # dataloading
        self.x = torch.from_numpy(numpy_array[:,:])
        self.num_samples = numpy_array.shape[0]


    def __getitem__(self, index):
        # dataset[0]
        return self.x[index].float()

    def __len__(self):
        # len(dataset)
        return self.num_samples

class Decoder_distribution():
    """Get distribution by sampling latent space of classial or quantum VAE"""

    def __init__(self, model, original_results:dict, nn_type:str, nsamples:int):
        self.model = model
        self.original_results = normalize(original_results) # set sum frequency = 1
        self.nsamples = nsamples
        self.nn_type = nn_type

    def get_outputstate_from_latentspace_CVAE(self) -> str:
        """For classical VAE: Sample from normal(0,1) from latent space and provide a single
        output bitstring"""
        
        with torch.no_grad():
            sample = torch.randn(model.z_mean.out_features) # inputs go through preprocessor
            output = self.model.decoding_fn(sample)
            p = torch.rand(output.shape) #np.random.rand(1)[0]
            state = torch.where(output>p, 1, 0)   
            bitstring = np.apply_along_axis("".join, 0, state.numpy().astype(int).astype(str))
        return bitstring

    def get_ouputdict_from_bitstrings(self, output_dict:dict, bitstring:str) -> dict:
        """If Bitstring is present in output dict then increase count by 1, 
        otherwise create new key"""
        if bitstring.item() in output_dict.keys():
            output_dict[bitstring.item()] += 1
        else:
            output_dict[bitstring.item()] = 1
            #print(bitstring.item())
        return output_dict

    def get_decoder_dist_QVAE(self):
        """For QVAE: Sample from latent space and return marginalised distribution"""
        nqubits = self.model.encoder[0].in_features
        outputs = np.zeros(( self.nsamples, pow(2, nqubits) ))
        with torch.no_grad():
            for idx, _ in enumerate(outputs):
                sample = torch.randn(self.model.z_mean.out_features)
                outputs[idx, :] = self.model.decoding_fn(sample)
        outputs = outputs.mean(axis=0) 
        bitstring_basis = [("{0:0%db}"%nqubits).format(i) for i in range(pow(2, nqubits))]

        return dict(zip(bitstring_basis, outputs))

    def get_fidelity(self, model_output_dict:dict) -> float:
        """Get Bhattacharya co-efficienct of two discrete distributions. 
        The two distributions are normalized (sum of frequencies set to 1) before computation"""
        fidelity_sqrt = 0
        for key in self.original_results.keys():
            if key in model_output_dict.keys():
                fidelity_sqrt += np.sqrt(self.original_results[key]*model_output_dict[key])
        return fidelity_sqrt**2

    def get_no_wrong_samples(self, model_output_dict:dict) -> float:
        """Get the number of wrong samples from the learnt distribution and the mass on right samples
        Wrong sample: bitstring that is not in original_results and has positive probability in learnt distribution"""
        wrong_samples = 0
        for basis_state in model_output_dict.keys():
            if (basis_state not in self.original_results) & (model_output_dict[basis_state] != 0):
                wrong_samples += 1 
        return wrong_samples

    def get_mass_rightsamples(self, model_output_dict:dict) -> float:
        """Get mass of right samples
        Sum of masses in learnt distribution that is on the basis seen in the original dist"""
        mass_right_samples = 0
        for basis_state in self.original_results:
            if basis_state in model_output_dict.keys():
                mass_right_samples += self.original_results[basis_state]
        return mass_right_samples

    def get_decoder_distribution(self):
        """Get distribution from decoder"""
        #output_states = np.zeros((self.nsamples, self.model.decoder[0].in_features))
        
        with torch.no_grad():
            if self.nn_type == "classical":
                output_dict = {}
                for i in range(self.nsamples):
                    bitstring = self.get_outputstate_from_latentspace_CVAE()
                    output_dict = self.get_ouputdict_from_bitstrings(output_dict, bitstring)
                
            elif self.nn_type == "quantum-classical":
                output_dict = self.get_decoder_dist_QVAE()
        
        output_dict = normalize(output_dict)
        no_wrong_samples = self.get_no_wrong_samples(output_dict)
        mass_right_samples = self.get_mass_rightsamples(output_dict)
        fidelity = self.get_fidelity(output_dict)
        return output_dict, no_wrong_samples, mass_right_samples, fidelity 

    def normalize(d, target=1.0):
        raw = sum(d.values())
        factor = target/raw
        return {key:value*factor for key,value in d.items()}


class Postprocessing():
    
    def __init__(self, current_dir:str, root_dir:str):
        self.current_dir = current_dir
        self.root_dir = root_dir
        
    def get_lossfilepaths(self):
        results = ['_training_data.txt','_valid_data.txt']
        loss_files = [os.path.join(self.current_dir,"%s%s"%(self.current_dir, fname)) for fname in results]
        loss_filepath = [os.path.join(self.root_dir, fname) for fname in loss_files]
        self.loss_filepath = loss_filepath
        
    def get_validlosses(self):
        self.get_lossfilepaths()
        df = pd.read_csv(self.loss_filepath[1], header=None, skiprows=3)
        df['valid_loss'] = df[0].apply(lambda x: x.split(' ')[1].split('(')[1][:-1]).astype(float)
        df.drop(0, axis=1,inplace=True)
        self.validlosses = df
        return df
    
    def get_trainingloss(self):
        self.get_lossfilepaths()
        training_losses = pd.read_csv(self.loss_filepath[0], skiprows=3, header=None)
        training_losses['combined_loss'] = training_losses[0].apply(lambda x: x.split(" ")[1][1:]).astype(float)
        training_losses['kl_loss'] = training_losses[2].apply(lambda x: x[:-2]).astype(float)
        training_losses['mse_loss'] = training_losses[1].astype(float)
        training_losses.drop([0,1,2], axis=1, inplace=True)
        self.training_losses = training_losses
        return training_losses
    
    def get_rightsample_masses(self):
        right_masses = "_mass_rightsamples.txt"
        rightmass_file = os.path.join(self.current_dir, "%s%s"%(self.current_dir, right_masses))
        rightmass_filepath = os.path.join(self.root_dir, rightmass_file)
        df = pd.read_csv(rightmass_filepath, skiprows=3, header=None)
        df['mass'] = df[0].apply(lambda x : x.split(' ')[1]).astype(float)
        df.drop(0, axis=1, inplace=True)
        self.rightsample_masses = df
        return df
    
    def get_output_dict(self):
        out_distfile_ext = "_output_dist.txt"
        out_dist_filepath = os.path.join(self.current_dir, "%s%s"%(self.current_dir, out_distfile_ext))
        out_dist_filepath = os.path.join(self.root_dir, out_dist_filepath)
    
        with open(out_dist_filepath, 'r') as rf:
            dist_file = rf.readlines()[3:]
        output_dictlist_epochwise = []
        for line in dist_file:
            given_dist = line.split("{")[1][:-3]
            dict_like = given_dist.split(', ')
            output_dict_epochwise = {}
            for j in dict_like:
                key = j.split('\'')[1]
                val = j.split(": ")[1]
                output_dict_epochwise[key]=val
            output_dictlist_epochwise.append(output_dict_epochwise)
        self.output_dictlist_epochwise = output_dictlist_epochwise
        return output_dictlist_epochwise
    
    def get_bhattacoef(self, distribtuion:dict, true_dist:dict) -> float:
        """Source:https://en.wikipedia.org/wiki/Fidelity_of_quantum_states"""
        fidelity_sqrt = 0
        for key in true_dist.keys():
            if key in distribtuion.keys():
                fidelity_sqrt += np.sqrt(float(distribtuion[key])*float(true_dist[key]))
        return fidelity_sqrt
        
    def get_fidelity_evolution(self, orig_results:dict):
        fidelity_evolution = []
        for dist_after_epoch in self.output_dictlist_epochwise:
            fidelity_evolution.append(self.get_bhattacoef(dist_after_epoch, orig_results)**2)
        return fidelity_evolution
 
        

   
class AutoPlotter():

    def __init__(self, root_dir:str, current_dir:str, early_stopper_obj, params:dict):
        self.root_dir = root_dir
        self.current_dir = current_dir
        self.early_stopper_obj = early_stopper_obj
        self.nqubits = params[0]; self.datafile =params[1]

    def plot_epochwise_lossfn(self, loss_list:list):
        
        train_loss, valid_losses, rightsample_masses = loss_list
        fig, axes = plt.subplots(1,5, figsize=(22,5))
        titles = "Combined_loss Mse_loss  Kl_loss Valid_loss Right_sample_mass".split()
        
        axes[0].plot(train_loss['combined_loss'].values, lw=3)
        axes[1].plot(train_loss['mse_loss'].values, lw=3)
        axes[2].plot(train_loss['kl_loss'].values, lw=3)
        axes[3].plot(valid_losses['valid_loss'].values, lw=3)
        axes[4].plot(rightsample_masses['mass'].values, lw=3)

        for idx, ax in enumerate(axes):
            ax.set_title(titles[idx], fontsize=18)
            ax.set_xlabel("No of epochs", fontsize=18)
            decorate_plot(ax)

        plt.tight_layout()
        plt.savefig("%s/training_loss_results.pdf"%(os.path.join(self.root_dir, self.current_dir)),
                    dpi=300,bbox_inches='tight')
        
    def plot_batchwise_lossfn(self, early_stopper_obj):
        
        fig, ax = plt.subplots(1,4, figsize=(22,5))
        titles = "Combined_loss Mse_loss Kl_loss Valid_loss Right_sample_mass".split()

        for idx, (key, value) in enumerate(early_stopper_obj.logdict.items()):
            ax[idx].plot(value, lw=3)
            ax[idx].set_title(titles[idx], fontsize=18)
            decorate_plot(ax[idx])
            ax[idx].set_xlabel("Epoch no", fontsize=18)
            if idx <3:
                ax[idx].plot(np.arange(63, len(value)), np.convolve(value, np.ones(64)/64, mode='valid'), lw=3)
                ax[idx].set_xlabel("Minibatch no", fontsize=18)

        plt.tight_layout()
        plt.savefig("%s/training_loss_results_minibatchwise.pdf"%(os.path.join(self.root_dir, self.current_dir)),
                    dpi=300,bbox_inches='tight')
        
    def plot_fidelity(self, fidelity_coefs):

        # Analyse distributions
        fig, ax = plt.subplots()
        decorate_plot(ax)
        ax.plot(fidelity_coefs, marker='o', ls="--")
        ax.set_xlabel("No epochs", fontsize=18)
        ax.set_ylabel("Fidelity", fontsize=18)
        ax.set_title("Fidelity with Bhattacharya coefficients", fontsize=18)
        plt.tight_layout()
        plt.savefig("%s/fidelity_results.pdf"%(os.path.join(self.root_dir, self.current_dir)),
                    dpi=300,bbox_inches='tight')
        

    def get_fidelity_coefs(self, postprocess):
        # Original results
        orig_results_normalized = normalize(get_dict_from_array(self.datafile))
        
        # Distribution obtained from model
        output_dictlist_epochwise = postprocess.get_output_dict()
        fidelity = postprocess.get_fidelity_evolution(orig_results_normalized)
        return fidelity

    def write_plots(self):
        """Analyse log files and write plots to current directory"""
        
        #model, dir_path, early_stopper_obj = trained_model
        postprocess = Postprocessing(self.current_dir, self.root_dir)
        train_losses = postprocess.get_trainingloss()
        valid_losses = postprocess.get_validlosses()
        rightsample_masses = postprocess.get_rightsample_masses()
        
        self.plot_epochwise_lossfn([train_losses, valid_losses, rightsample_masses])
        self.plot_batchwise_lossfn(self.early_stopper_obj)
        fidelity_coefs = self.get_fidelity_coefs(postprocess)
        print(np.array(fidelity_coefs))
        np.savetxt(os.path.join(self.root_dir, self.current_dir+"/fidelity_evolution.txt"), fidelity_coefs)
        self.plot_fidelity(fidelity_coefs)