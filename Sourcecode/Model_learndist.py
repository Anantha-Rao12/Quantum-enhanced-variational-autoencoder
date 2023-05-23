import torch
import qiskit
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import CircuitQNN

#######################################################################
# Setup quantum circuit decoder
#######################################################################

def create_qnn(num_inputs:int, num_qubits:int, qc_params:dict):
    """Creates the decoder circuit with ansatz. 

	Parameters:
	num_inputs (int) : size of the input vector that is embedded to the quantum circuit
	num_qubits (int) : number of qubits in the circuit
	qc_params (dict) : dictionary that specifies the feature map (currently only Pauli ZZ, Z, P), entanglement type and the number of repetiton layers.

	Returns:
	A quantum neural network object, and the designed quantum circuit."""
    
    if num_inputs > num_qubits:
        raise ValueError('Number of inputs is greater than the number of qubits... Not suitable with current feature map')
    
    fm , entanglement_type, repititions = qc_params.values()
    
    if fm == 'ZZ':
        feature_map = qiskit.circuit.library.ZZFeatureMap(num_inputs)
    elif fm == 'Z':
        feature_map = qiskit.circuit.library.ZFeatureMap(num_inputs)
    elif fm == 'P':
        feature_map = qiskit.circuit.library.PauliFeatureMap(num_inputs, reps=2, paulis=['X', 'Y'], insert_barriers=True)
    else:
        raise ValueError("Wrong feature Map provided!")

    #local_entanglement = {}
    ansatz = qiskit.circuit.library.TwoLocal(
        num_qubits=num_qubits,
        rotation_blocks=["ry", "rx"],
        entanglement_blocks='cx',
        skip_final_rotation_layer=False,
        entanglement=entanglement_type,
        reps=repititions, #1
        insert_barriers=True,
    )
    qc = qiskit.QuantumCircuit(num_qubits)
    qc.append(feature_map, range(0, num_inputs))
    qc.h(range(num_inputs, num_qubits)) if num_inputs < num_qubits else None
    qc.barrier()
    qc.append(ansatz, range(num_qubits))
    qnn = CircuitQNN(
        qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        input_gradients=True,
        sparse=False,
        quantum_instance=qiskit.Aer.get_backend("qasm_simulator"),
    )
    return qnn, qc


class QVAE_qcompile(torch.nn.Module):
	"""
	Create the hybrid quantum classical neural network

	Parameters:
	qnn : qiskit quantum neural network object
	latent_dim (int) : Latent dimension of the model

	Returns:
	A pytorch neural network object
	"""

    def __init__(self, qnn, latent_dim:int):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(qnn.circuit.num_qubits, 8),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(8, 7),
            torch.nn.LeakyReLU(0.01)) 

        self.z_mean = torch.nn.Linear(7,latent_dim)
        self.z_log_var = torch.nn.Linear(7,latent_dim)

        # self.preprocessor = torch.nn.Sequential(
        #     torch.nn.Linear(latent_dim, qnn.circuit.num_qubits),
        #     torch.nn.LeakyReLU(0.01))
        
        self.preprocessor = torch.nn.Linear(latent_dim, qnn.circuit.num_qubits)
        torch.nn.init.normal_(self.preprocessor.weight, mean=0, std=0.01)
        torch.nn.init.constant_(self.preprocessor.bias, val=0)
            
        self.decoder = TorchConnector(qnn)

    def encoding_fn(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return encoded
        
    def decoding_fn(self, x):
        x = self.preprocessor(x)
        decoded = self.decoder(x)
        return decoded
        
    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1))
        z = z_mu + eps * torch.exp(z_log_var/2.) 
        return z
        
    def forward(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        preprocessed = self.preprocessor(encoded)
        decoded = self.decoder(preprocessed)
        return encoded, z_mean, z_log_var, decoded #preprocessed

