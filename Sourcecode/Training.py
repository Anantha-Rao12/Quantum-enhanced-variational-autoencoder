import os, time
import math
from datetime import datetime
import datetime
import numpy as np
import torch

from Auxillary_functions import MeasurementDataset, now, Decoder_distribution
  
                
class QeVAE():
	"""
	Creates a QeVAE object that is ready for training. 

	Parameters:
	training_params (dict) : patience factor, minibatchsize, beta, annealing schedule
	dataloaders (dict) : Training and validation set pytorch dataloader objects
	optimizers (dict) : Pytorch optimizers for the encoder and decoder
	true_result_keys (list) : True distribution obtained from data
	root_dir (str) : Directory where results are to be written
	nn_type (str) : Type of the neural network (classical/quantum-classical)
	"""
    
    def __init__(self, training_params:dict, dataloaders:list,
                optimizers:list, true_result_keys:list, root_dir:str, nn_type:str):

        self.patience, self.minibatchsize, self.beta, self.annealing_schedule = training_params.values()
        self.train_dataloader, self.valid_dataloader = dataloaders
        self.optimizer_classical, self.optimizer_quantum  = optimizers
        self.true_result_keys = true_result_keys
        self.root_directory = root_dir
        self.nn_type = nn_type

        self.decoder_params_list = []
        self.no_wrong_samples = []
        self.right_samples_mass = []
        self.output_dict_list = []
        self.fidelity = []
    
        self.logdict =  {'train_combined_loss_per_minibatch': [],
                    'train_reconstruction_loss_per_minibatch': [],
                    'train_kl_loss_per_minibatch': [],
                   'valid_reconstruction_loss':[]}


    def get_lr(self):
        """Gives the learning rate of the optimizers"""
        for param_group in self.optimizer_classical.param_groups:
            classical_lr =  str(param_group["lr"]).split(".")[-1]
        for param_group in self.optimizer_quantum.param_groups:
            quantum_lr =  str(param_group["lr"]).split(".")[-1]
            
        return "%s_%s"%(classical_lr, quantum_lr)

    def get_datafiles(self):
        """Given a root directory and the particular classical optimizer, creates files to store
        the training and valid scores, encoder parameters during training"""

        rightnow = now(0)
        fname = "qvae_qstatecompilation_lr%s_%s" % (self.get_lr(), rightnow)
        dir_path = os.path.join(self.root_directory, fname)
        os.mkdir(dir_path)

        training_datafilename = os.path.join(dir_path, fname + "_training_data.txt")
        valid_datafilename = os.path.join(dir_path, fname + "_valid_data.txt")
        encoder_params_datafilename = os.path.join(dir_path, fname + "_encoder_data.txt")
        decoder_params_datafilename = os.path.join(dir_path, fname + "_decoder_data.txt")
        mass_rightsamples_filename = os.path.join(dir_path, fname + "_mass_rightsamples.txt")
        output_dist_filename = os.path.join(dir_path, fname+"_output_dist.txt")
        fidelity_dist_filename = os.path.join(dir_path, fname+"_fidelity.txt")


        filenames = [
            training_datafilename, valid_datafilename,
            encoder_params_datafilename, decoder_params_datafilename,
            mass_rightsamples_filename, output_dist_filename,
            fidelity_dist_filename]
            
        firstlines = [
            "Training data goes here...",
            "Validation data goes here...",
            "Encoder parameters go here...",
            "Decoder parameters go here...",
            "Mass on desired states go here...",
            "Output distribtuion from decoder goes here...",
            "Fidelity data goes here..."
        ]

        for idx, filename in enumerate(filenames):
            with open(filename, "a") as append_file:
                append_file.write(
                    "%s\nDate:%s\nAuthor:Anantha Rao\n" % (now(0), firstlines[idx])
                )

        return filenames, dir_path

    def push_data2file(self, epoch_no: int, data: list, datafiles: list):
        """Data is a list of values which need to be written to a particuar file given by the same index in datafiles
        In data[0]: Training data
           data[1]: Validation data
            data[2]: Encoder parameters
           data[3]: Decoder parameters
           data[4]: Mass on right samples 
        In data[0] the entries represent the
            average epoch total training loss, training MSE, training KL divergence and
            validation scores. These are flused into the respective files after each epoch"""

        for counter, fname in enumerate(datafiles):
            with open(fname, "a") as append_file:
                append_file.write("%s %s \n" % (epoch_no, data[counter]))
                
                
    def get_model_info(self, model, num_epochs:int):
        """Returns the parameters of the model"""
        # Print number of trainable parameters
        encoder_features = [model.encoder, model.z_mean, model.z_log_var]
        encoder_trainable_params = sum(list(map(lambda x : sum(p.numel() for p in x.parameters()), encoder_features)))
        total_params = sum(p.numel() for p in model.parameters())
        
        trainparams_info = "No of trainable parameters: \n (Model:%d) | (Encoder:%d) | (Decoder:%d)"%(total_params, encoder_trainable_params, total_params-encoder_trainable_params)
        dataloader_info = "Size of training set: %d | Size of validation set: %d"%(len(self.train_dataloader), len(self.valid_dataloader))
        epoch_batch_info = "Total number of epochs: %d | Total number of batches: %d"% (num_epochs, math.ceil(len(self.train_dataloader)/self.minibatchsize))
        latentsize = "Latent size: %s"%int(model.z_mean.out_features)
        batchsize_info = "Batchsize: %d"%(self.minibatchsize)
        kl_term_weight = "KL Term weight: %f ; Type: %s"%(self.beta, self.annealing_schedule)
        return trainparams_info, dataloader_info, epoch_batch_info, latentsize, batchsize_info, kl_term_weight
        
    def validation_loss(self, model, all_bitstrings):
	"""Computes Validation loss using only the reconstruction term"""

        output_qubits = model.encoder[0].in_features    
        total_val_loss = 0

        with torch.no_grad():
            for idx, data in enumerate(self.valid_dataloader):
                encoded, z_mean, z_log_var, decoded = model(data)
                
                if self.nn_type == "quantum-classical":
                    input_bitstring = np.apply_along_axis(''.join, 1, data.detach().numpy().astype(int).astype(str))
                    meas_dict = dict(zip(all_bitstrings, (decoded.squeeze()+pow(2,-18))/( 1+pow(2, -18)*pow(2, output_qubits) ) ))
                    likelihood_losses = -torch.log(meas_dict[input_bitstring[0]])
                    total_val_loss += likelihood_losses
                
                elif self.nn_type =='classical':
                    likelihood_losses = torch.nn.BCELoss()(decoded, data) #torch.nn.CrossEntropyLoss()
                    total_val_loss += likelihood_losses

            return total_val_loss/len(self.valid_dataloader)

            
    def start_training(self, model, all_bitstrings:list, num_epochs:int, original_results:dict):
	""" Starts training the QeVAE model and logs output of training after every minibatch

	Parameters:
	model : Pytorch model 
	all_bitstrings (list) : All 2^n bitstrings where n is the number of qubits
	num_epochs (int) : Total number of epochs for training
	original_results (dict) : True results from data including data and probability

	Returns:
	Trained mode, output directory path
	"""
        
        output_qubits = model.encoder[0].in_features
        
        trigger_times = 0 ; trigger_vals=[]
        filenames, directory_path = self.get_datafiles()
        print("Directory path:", directory_path)
        with open(os.path.join(directory_path, "model_info.txt"), 'a') as wf:
            outputtext = self.get_model_info(model, num_epochs)
            for i in range(len(outputtext)):
                wf.write("%s\n"%outputtext[i])
            
        loss_list = []
        
        # Early stopping
        last_loss = 20; init_mass_rightsamples = 0; trigger_times = 0

        for epoch in range(num_epochs):
            total_loss = []; epoch_kl_loss = []; epoch_mse_loss = []
            kl_term_weight = self.get_kl_term_weight(epoch, num_epochs, self.annealing_schedule)
            minibatchsize_no = 0
            start_time = time.time()
            
            for batch_idx, data in enumerate(self.train_dataloader):

                # Forward pass
                encoded, z_mean, z_log_var, decoded = model(data)

                if self.nn_type == "quantum-classical":
                    input_bitstring = np.apply_along_axis(''.join, 1, data.numpy().astype(int).astype(str))
                    measurement_dict = dict(zip(all_bitstrings, (decoded.squeeze()+pow(2,-18))/( 1+pow(2,-18)*pow(2,output_qubits) ) ))
                    likelihood_losses = -torch.log(measurement_dict[input_bitstring[0]])

                if self.nn_type == "classical":
                    likelihood_losses = torch.nn.BCELoss()(decoded, data) #torch.nn.CrossEntropyLoss()

                # Normalize loss for batch accumulation
                mean_ll = likelihood_losses/self.minibatchsize
                kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean**2 - torch.exp(z_log_var))
                kl_loss = (kl_term_weight*kl_loss)/self.minibatchsize
                loss = mean_ll + kl_loss

                # Backward pass
                loss.backward()  

                # weights update
                if  ((batch_idx + 1) % self.minibatchsize == 0) or (batch_idx + 1 == len(self.train_dataloader)):
                    minibatchsize_no += 1
                    self.optimizer_classical.step(); self.optimizer_quantum.step()  # Optimize weights
                    self.optimizer_classical.zero_grad(set_to_none=True)  # reset gradients to zero
                    self.optimizer_quantum.zero_grad(set_to_none=True)  # reset gradients to zero

                    # Store losses
                    total_loss.append(loss.item())  
                    epoch_mse_loss.append(mean_ll.item())
                    epoch_kl_loss.append(kl_loss.item())

                    print("(%d|%d , %d|%d) Total loss: %.5f | Likelihood loss: %.5f | KL loss : %.5f "
                    %(epoch+1, num_epochs, minibatchsize_no, math.ceil(len(self.train_dataloader)/self.minibatchsize), loss.item(),
                    mean_ll.item(), kl_loss.item()))

                    # LOGGING
                    self.logdict['train_combined_loss_per_minibatch'].append(loss.item())
                    self.logdict['train_reconstruction_loss_per_minibatch'].append(mean_ll.item())
                    self.logdict['train_kl_loss_per_minibatch'].append(kl_loss.item())

            end_time = time.time()
            valid_loss = self.validation_loss(model, all_bitstrings)
            self.logdict['valid_reconstruction_loss'].append(valid_loss) 

            # Store params
            if self.nn_type == "quantum-classical":
                decoder_params = model.decoder.weight.detach().numpy().copy()
            if self.nn_type == "classical":
                decoder_params = [param.detach().numpy().tolist() for param in model.decoder.parameters()]
                
            encoder_params = [param.detach().numpy().tolist() for param in model.encoder.parameters()]
            self.decoder_params_list.append(decoder_params)
            
            output_dist_obj = Decoder_distribution(model, original_results, self.nn_type, nsamples=5000)
            output_dict, n_wrong_samples, mass_right_samples, fidelity = output_dist_obj.get_decoder_distribution()
            
            self.no_wrong_samples.append(n_wrong_samples); self.right_samples_mass.append(mass_right_samples)
            self.output_dict_list.append(output_dict); self.fidelity.append(fidelity)
            print("Fidelity:", fidelity)
            
            # Push latest data to file
            training_lossdata = list(map(lambda x: sum(x)/len(x), [total_loss, epoch_mse_loss, epoch_kl_loss] ))
            data_list = [training_lossdata, valid_loss, encoder_params, decoder_params, mass_right_samples, output_dict, fidelity]
            self.push_data2file(epoch, data_list, filenames)

            # Store loss for epoch
            loss_list.append(sum(total_loss) / len(total_loss))

            print("Time taken %4fs"%(end_time - start_time))
            print('Epoch: %02d/%02d | Beta %.3f | Avg Train Loss: %.4f | Valid Loss: %.4f | Wrong states: %d | Mass on right states %.4f\n'
                        % (epoch+1, num_epochs, kl_term_weight, loss_list[-1],valid_loss, n_wrong_samples, mass_right_samples))

            torch.save(model.state_dict(),
                os.path.join(directory_path, "epoch_%d.pth" % epoch)
            )

            if valid_loss > last_loss: #mass_right_samples < init_mass_rightsamples:
                trigger_times += 1
                trigger_vals.append(trigger_times)
                print(f"Trigger Times: {trigger_times} \n")

                if trigger_times >= self.patience:
                    print("Early stopping! Closing training. Now can start to test process.")
                    with open(os.path.join(directory_path,'logged_data.txt'), 'a') as dump_logfile:
                        for key, value in self.logdict.items():
                            dump_logfile.write('%s:%s\n' % (key, value))
                            dump_logfile.write('\n')
                    return model, directory_path

            elif valid_loss <= last_loss: #mass_right_samples >= init_mass_rightsamples:
                print("Trigger times: 0\n")
                trigger_times = 0
                last_loss = valid_loss
                #init_mass_rightsamples = mass_right_samples

        with open(os.path.join(directory_path,'logged_data.txt'), 'a') as dump_logfile:
            for key, value in self.logdict.items():
                dump_logfile.write('%s:%s\n' % (key, value))
                dump_logfile.write('\n')
        return model, directory_path
        
    def get_kl_term_weight(self, epoch:int, num_epochs:int, schedule:str):
        """Returns a kl term based on predefined annealing schedule"""
        
        if schedule =='linear':
            return (epoch / num_epochs) * self.beta
            
        if schedule == 'fixed':
            return self.beta
        
        if schedule == 'zero':
            return 0
            
        if schedule == 'stepfn':
            if epoch <= 35:
                return self.get_kl_term_weight(epoch, num_epochs, schedule='zero')
                
            else:
                return self.get_kl_term_weight(epoch, num_epochs, schedule='fixed')
                
        if schedule =='stepfn_linear':
            if epoch <= 30:
                return self.get_kl_term_weight(epoch, num_epochs, schedule='zero')
                
            else:
                return self.get_kl_term_weight(epoch-30, num_epochs, schedule='linear')