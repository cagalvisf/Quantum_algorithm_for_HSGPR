import numpy as np
import GP_regressors

"""
Here we implement the necessary functions to build the circuit for Quantum Gaussian Process Regression 
usin the Hilbert Space approximation of the kernel.
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, transpile, Aer
from qiskit.circuit.library import QFT
from qiskit.circuit.library.standard_gates import RYGate
from qiskit.circuit.library.data_preparation.state_preparation import StatePreparation
from qiskit.extensions import HamiltonianGate
from qiskit.quantum_info import Statevector as st

import scipy

from operator import itemgetter

backend = Aer.get_backend("aer_simulator")

## compare the similitud of two binary strings
def compare_binary_strings(bin1, bin2):
    count = 0
    for i in range(len(bin1)):
        if (bin1[i]!= bin2[i]):
            break
        if (bin1[i]=="1"):
            count += 1
    return count

## Cleaning the repetitive eigenvalues
def clean_binary(list_bin):
    for i in range(0, len(list_bin)):
        for j in range(i+1, len(list_bin)):
            if (compare_binary_strings(list_bin[i], list_bin[j]) >= 1):
                list_bin[j] = str(0)*len(list_bin[0])
            if (list_bin[i] == str(0)*len(list_bin[0])):
                break

    list_bin = [i for i in list_bin if i != str("0")*len(list_bin[0])]

    if (list_bin[-1] == str("0")*(len(list_bin[0])-1)+"1") and (list_bin[-2] == str("0")*(len(list_bin[0])-2) + "10"):
        list_bin = list_bin[:-1]

    
    return list_bin

## Function to compute the main counts
def main_counts(counts, n_eig):
    ## save the counts before str(0)*n_eig in a dictionary
    counts_main = {}
    for k in counts.keys():
        if k[:n_eig] != str(0)*n_eig:
            counts_main[k] = counts[k]
        else:
            break
    
    return counts_main

## Function to compute the eigenvalues
def quantum_eigenvals(Z,
                    M,
                    n_eig,
                    delta,
                    shots):
    
    ZZd = Z.T @ Z

    psi = Z.flatten()
    psi_vector = st(psi)
    psi_gate = StatePreparation(psi_vector)

    ## Create the circuit
    n_psi = int(np.log2(len(psi)))
    T = 2*np.pi/delta
    

    ## Creation of quantum registers
    psi_1 = QuantumRegister(n_psi, "\psi")
    eigen = QuantumRegister(n_eig, "\kappa")



    ## Circuit to estimate eigenvalues
    QPE_circuit = QuantumCircuit(eigen,psi_1)

    ## Superposition of the lambda qubits
    for i in range(n_eig):
        QPE_circuit.h(eigen[i])

    ## Evolution operatot for Z.T @ Z
    U = HamiltonianGate(-ZZd, T)
    U.name = "U"

    ## Controlled operations for quantum phase estimation
    for i in range(n_eig):
        Upow = U.power(2**(i)) 
        ctrl_Upow = Upow.control(1)
        C_qubit = eigen[i]
        
        ## Controlled evolution operator over the last M qubits
        QPE_circuit.append(ctrl_Upow, [C_qubit]+psi_1[:int(np.log2(M))])


    ## Quantum fourier transform 
    qfti_gate = QFT(num_qubits= n_eig, inverse=True).to_gate()
    QPE_circuit.append(qfti_gate, eigen)
    

    ## Circuit to measure the eigenvalues
    ## Creation of classical registers
    cr = ClassicalRegister(n_eig,'c')
    ## Optimization of the circuit

    #transpiled_circuit = transpile(QPE_circuit, backend=backend, optimization_level=0)
    #QPE_circuit = transpiled_circuit

    ## Circuit intialization
    circuit_measure = QuantumCircuit(eigen, psi_1, cr)
    circuit_measure.append(psi_gate, psi_1)
    
    ## QPE circuit
    circuit_measure.append(QPE_circuit, eigen[:]+psi_1[:])
    circuit_measure.barrier()
        

    for i in range(n_eig):
        circuit_measure.measure(eigen[i],cr[i])

    job = execute(circuit_measure, backend, shots=shots)
    result = job.result()
    counts = result.get_counts(circuit_measure)

    sort_counts= dict(sorted(counts.items(), key=itemgetter(1), reverse=True))

    
    ## main counts
    counts_main = main_counts(sort_counts, n_eig)

    theta_bin=[*counts_main.keys()]
    theta_bin = clean_binary(theta_bin)

    ## if lenght of theta_bin bigger than M, just take the first M elements
    if len(theta_bin) > M:
        theta_bin = theta_bin[:M]
        

    eigenvals_quantum = []
    for j in range(0,len(theta_bin)):
        
        zeta=(int(theta_bin[j],2)/(2**n_eig))*(2*np.pi/T)
        
        eigenvals_quantum.append(zeta)
        
        #print("Eigenvalue qPU: ", zeta)
        #print("C:", C)
        #print("infint: ",2*np.arcsin(C/(zeta)))
    eigenvals_quantum = np.array(eigenvals_quantum)
        
        #print("Eigenvalue qPU: ", zeta)
    ## create a dictionary with the binary strings as key and the eigenvalues as values
    eigenvals_dict = dict(zip(theta_bin, eigenvals_quantum))
    ## sort the dictionary by the eigenvalues in ascending order
    eigenvals_dict = dict(sorted(eigenvals_dict.items(), key=itemgetter(1), reverse=True))

    return eigenvals_dict, QPE_circuit

## Function to compute the mean of the posterior distribution
def mean_QGPR_approximation_posterior(Phisf, Y, Lambda, Z, sigma2, q_eigen_dict, QPE_circuit, n_eig, R, shots=2**5):
    backend = Aer.get_backend("aer_simulator")
    Zs =np.array(Phisf.T @ np.sqrt(Lambda))

    ## Mean calculation
    psi_1 = Z.flatten()
    psi_1_vector = st(psi_1)
    psi_1_gate = StatePreparation(psi_1_vector)

    ## Create the circuit
    n_psi = int(np.log2(len(psi_1)))


    ## Creation of quantum registers
    ancilla = QuantumRegister(1, "a")
    psi_1_qr = QuantumRegister(n_psi, "\psi")
    eigen_qr = QuantumRegister(n_eig, "\kappa")

    qc_psi_1 = QuantumCircuit(ancilla, eigen_qr, psi_1_qr)

    ## initialize the psi register in the state psi
    qc_psi_1.append(psi_1_gate, psi_1_qr)

    ## Apply QPE_ciruit to the registers psi_1 and eigen
    qc_psi_1.append(QPE_circuit, eigen_qr[:] + psi_1_qr[:])

    ## unpack the eigenvalues and the binary strings from the dictionary
    theta_bin = [*q_eigen_dict.keys()]
    quantum_eigvals = [*q_eigen_dict.values()]

    C = quantum_eigvals[R-1] + sigma2 

    for j in range(0,R):
        binary_string =theta_bin[j]
        count_ones = binary_string.count("1")

        zeta=quantum_eigvals[j]

        #print(zeta)
        #print(C/(zeta + sigma2))

        ##one_positions = [i for i, digit in enumerate(binary_string[::-1]) if digit == "1"]
        one_positions = [i for i, digit in enumerate(binary_string[::-1]) if digit == "1"]
        ##print(one_positions)
        cry=RYGate(2*np.arcsin(C/(zeta + sigma2))).control(count_ones)

        control_qubits = [eigen_qr[i] for i in one_positions]
        target_qubit = ancilla
        qc_psi_1.append(cry, control_qubits+[target_qubit] )

    qc_psi_1.append(QPE_circuit.inverse(), eigen_qr[:] + psi_1_qr[:])

    ## optimization of the circuit
    #transpiled_circuit = transpile(qc_psi_1, backend=backend, optimization_level=0)
    #qc_psi_1 = transpiled_circuit

    Hadamard_circuit1 = qc_psi_1.decompose().to_gate().control(1, ctrl_state='0')

    Phi2=np.kron(Y, Zs)
    Phi2_norm=np.linalg.norm(Phi2)
    Phi2=Phi2/Phi2_norm
    PSI2 = Phi2.flatten()

    phis_y_vector=st(PSI2)
    phis_y_gate = StatePreparation(phis_y_vector)

    ancilla_2 = QuantumRegister(1, "a")
    eigen_2_qr = QuantumRegister(n_eig, "\kappa")
    psi_2_qr = QuantumRegister(n_psi, "\psi")

    qc_2 = QuantumCircuit(ancilla_2,eigen_2_qr, psi_2_qr)

    qc_2.append(phis_y_gate, psi_2_qr)
    qc_2.x(ancilla_2)

    Hadamard_circuit2 = qc_2.decompose().to_gate().control(1, ctrl_state='1')

    ## hadamard test
    psi_h = QuantumRegister(n_psi, "\psi")
    eigen_h = QuantumRegister(n_eig, "\kappa")
    ancilla_h = QuantumRegister(1, "a")
    ancilla_h2 = QuantumRegister(1, "a2")

    cr = ClassicalRegister(1, "c")

    qc_h = QuantumCircuit(ancilla_h, eigen_h, psi_h, ancilla_h2, cr)

    qc_h.h(ancilla_h2)
    qc_h.append(Hadamard_circuit1, ancilla_h2[:] + ancilla_h[:] + eigen_h[:] + psi_h[:])
    qc_h.append(Hadamard_circuit2, ancilla_h2[:] + ancilla_h[:] + eigen_h[:] + psi_h[:])
    qc_h.h(ancilla_h2)

    ## optimization of the circuit
    #transpiled_circuit = transpile(qc_h, backend=backend, optimization_level=3)
    #qc_h = transpiled_circuit

    qc_h.measure(ancilla_h2, cr)

     
    backend = Aer.get_backend("aer_simulator")
    job_ht = execute(qc_h, backend, shots=shots)
    result_ht = job_ht.result()
    counts_ht = result_ht.get_counts(qc_h)

    Exp = 0
    print(counts_ht)
    prob_0 = counts_ht['0']/shots 
    Exp = 2*prob_0 - 1
    mean_sim = (Exp/C)*Phi2_norm

    return mean_sim

def var_QGPR_approximation_posterior(Phisf, Lambda, Z, sigma2, q_eigen_dict, QPE_circuit, n_eig, R, M, shots=2**14):
    
    Zs =np.array(Phisf.T @ np.sqrt(Lambda))
    theta_bin = [*q_eigen_dict.keys()]
    quantum_eigvals = [*q_eigen_dict.values()]

    ## Mean calculation
    psi_1 = Z.flatten()
    psi_1_vector = st(psi_1)
    psi_1_gate = StatePreparation(psi_1_vector)

    ## Create the circuit
    n_psi = int(np.log2(len(psi_1)))


    ## Creation of quantum registers
    ancilla_r = QuantumRegister(1, "a_r")
    ancilla_s = QuantumRegister(1, "a_s")
    psi_1_qr = QuantumRegister(n_psi, "\psi_1")
    eigen_qr = QuantumRegister(n_eig, "\kappa")

    cr_r = ClassicalRegister(1, "c_r")
    cr_s = ClassicalRegister(1, "c_s")
    

    Phi2=Zs.T
    Phi2_norm=np.linalg.norm(Phi2)
    Phi2=Phi2/Phi2_norm
    PSI2 = Phi2.flatten()

    phis_y_vector=st(PSI2)
    phis_y_gate = StatePreparation(phis_y_vector)

    n_psi_2 = int(np.log2(len(PSI2)))
    psi_2_qr = QuantumRegister(n_psi_2, "\psi_2")
    

    qc_swap = QuantumCircuit(ancilla_r, eigen_qr, psi_1_qr, psi_2_qr, ancilla_s, cr_r, cr_s)

    ## initialize the psi register in the state psi
    qc_swap.append(psi_1_gate, psi_1_qr)
    qc_swap.append(phis_y_gate, psi_2_qr)
    ## Apply QPE_ciruit to the registers psi_1 and eigen
    qc_swap.append(QPE_circuit, eigen_qr[:] + psi_1_qr[:])

    C = np.sqrt(quantum_eigvals[R-1]*(quantum_eigvals[R-1] + sigma2))
    P_1=0
    for j in range(0,R):
        binary_string =theta_bin[j]
        count_ones = binary_string.count("1")

        zeta=quantum_eigvals[j]
        P_1=P_1+(1/(zeta+sigma2))
        #print(zeta)
        #print(C/(zeta + sigma2))

        one_positions = [i for i, digit in enumerate(binary_string[::-1]) if digit == "1"]
        cry=RYGate(2*np.arcsin(C/np.sqrt(zeta*(zeta + sigma2)))).control(count_ones)

        control_qubits = [eigen_qr[i] for i in one_positions]
        target_qubit = ancilla_r
        qc_swap.append(cry, control_qubits+[target_qubit] )

    

    qc_swap.append(QPE_circuit.inverse(), eigen_qr[:] + psi_1_qr[:])

    qc_swap.h(ancilla_s)
    for i in range(0, n_psi_2):
        qc_swap.cswap(ancilla_s, psi_1_qr[i], psi_2_qr[i])
    qc_swap.h(ancilla_s)

    qc_swap.measure(ancilla_r, cr_r)
    qc_swap.measure(ancilla_s, cr_s)

    backend = Aer.get_backend("aer_simulator")
    job_st = execute(qc_swap, backend, shots=shots)
    result_st = job_st.result()
    counts_st = result_st.get_counts(qc_swap)
    
    print(counts_st)

    prob_11 = counts_st['1 1']/shots 
    var = (P_1 - 2*prob_11/C**2)*Phi2_norm**2
    return var ,qc_swap

def QGPR_approximation_posterior(Xp, mean_args):
    X = mean_args[0]
    Y = mean_args[1]
    sigma2 = mean_args[2]
    M = mean_args[3]
    L = mean_args[4]
    alpha = mean_args[5]
    scale = mean_args[6]
    delta = mean_args[7]
    n_eig = mean_args[8]
    R = mean_args[9]
    shots = mean_args[10]

    HS = GP_regressors.HS_approx_GPR((X, Y),
                   sigma2=sigma2,
                   M = M,
                   L = L,
                   alpha=alpha,
                   scale=scale)

    Phif = HS.Phi_matrix(X)
    Lambda = HS.Lambda()
    
    Z = np.array(Phif @ np.sqrt(Lambda))

    ## Normalize the data
    norm_z = np.linalg.norm(Z)

    Z_norm = Z/norm_z

    ZZd = np.array(Z_norm.T @ Z_norm)

    ## For demonstration, we calculate the eigenvalues of ZZd and chose delta based on that
    real_eigenvals, real_eigenvecs = scipy.linalg.eig(ZZd)

    ## delta parameter should be 1>delta>lam_max
    q_eigen_dict, QPE_circuit = quantum_eigenvals(Z_norm, M, n_eig, delta, shots)

    print("Real eigenvalues: ", real_eigenvals)
    print("Quantum eigenvalues: ", [*q_eigen_dict.values()])


    mu = []
    var = []
    i=0
    ## Parameters
    for xp in Xp:
        Phisf = HS.Phi_matrix([xp]).T
                
        mu.append(mean_QGPR_approximation_posterior(Phisf, Y, Lambda, Z_norm, sigma2, q_eigen_dict, QPE_circuit, n_eig, R,  shots))

        v, v_qc = var_QGPR_approximation_posterior(Phisf, Lambda, Z_norm, sigma2, q_eigen_dict, QPE_circuit, n_eig, R, M, shots)
        var.append(np.abs(v))
        i+=1
        print("Point: ", i, " of ", len(Xp), " done.")
    return np.array(mu)/norm_z, sigma2*np.array(var)/norm_z**2, v_qc
