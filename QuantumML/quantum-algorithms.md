# Quantum Algorithms for Machine Learning

Deep dive into quantum algorithms that power quantum machine learning applications.

**Last Updated:** 2025-06-19

## Table of Contents
- [Foundational Quantum Algorithms](#foundational-quantum-algorithms)
- [Quantum Search & Optimization](#quantum-search--optimization)
- [Quantum Linear Algebra](#quantum-linear-algebra)
- [Variational Quantum Algorithms](#variational-quantum-algorithms)
- [Quantum Sampling Algorithms](#quantum-sampling-algorithms)
- [Hybrid Classical-Quantum Algorithms](#hybrid-classical-quantum-algorithms)
- [Implementation Examples](#implementation-examples)
- [Performance Analysis](#performance-analysis)

## Foundational Quantum Algorithms

### Deutsch-Jozsa Algorithm
**Purpose**: Determine if function is constant or balanced
- **Speedup**: Exponential over classical
- **Complexity**: O(1) vs O(2^n)
- **Significance**: First quantum advantage proof

```python
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

def deutsch_jozsa(oracle, n_qubits):
    qr = QuantumRegister(n_qubits + 1)
    cr = ClassicalRegister(n_qubits)
    qc = QuantumCircuit(qr, cr)
    
    # Initialize
    qc.x(qr[-1])  # Ancilla in |1⟩
    qc.h(qr)      # Hadamard all qubits
    
    # Oracle
    qc.append(oracle, qr)
    
    # Hadamard on input qubits
    for i in range(n_qubits):
        qc.h(qr[i])
    
    # Measure
    qc.measure(qr[:-1], cr)
    
    return qc
```

### Bernstein-Vazirani Algorithm
**Purpose**: Find hidden bit string
- **Application**: Feature extraction
- **Advantage**: Single query vs n queries
- **ML Use**: Pattern recognition

### Simon's Algorithm
**Purpose**: Find hidden period
- **Speedup**: Exponential
- **Application**: Cryptanalysis
- **ML Connection**: Periodicity detection

## Quantum Search & Optimization

### Grover's Algorithm
**Unstructured Search:**
```python
def grover_oracle(marked_item, n_qubits):
    """Create oracle for Grover's algorithm"""
    oracle = QuantumCircuit(n_qubits)
    
    # Flip phase of marked item
    # Convert marked_item to binary
    for i, bit in enumerate(format(marked_item, f'0{n_qubits}b')):
        if bit == '0':
            oracle.x(i)
    
    # Multi-controlled Z gate
    oracle.h(n_qubits-1)
    oracle.mcx(list(range(n_qubits-1)), n_qubits-1)
    oracle.h(n_qubits-1)
    
    # Undo X gates
    for i, bit in enumerate(format(marked_item, f'0{n_qubits}b')):
        if bit == '0':
            oracle.x(i)
    
    return oracle

def grover_diffuser(n_qubits):
    """Inversion about average"""
    qc = QuantumCircuit(n_qubits)
    
    # H gates
    qc.h(range(n_qubits))
    
    # X gates
    qc.x(range(n_qubits))
    
    # Multi-controlled Z
    qc.h(n_qubits-1)
    qc.mcx(list(range(n_qubits-1)), n_qubits-1)
    qc.h(n_qubits-1)
    
    # X gates
    qc.x(range(n_qubits))
    
    # H gates
    qc.h(range(n_qubits))
    
    return qc
```

**Applications in ML:**
- Database search
- Optimization problems
- Feature selection
- Hyperparameter search

### Quantum Approximate Optimization Algorithm (QAOA)
**Combinatorial Optimization:**
```python
from qiskit.circuit.library import QAOAAnsatz
from qiskit_optimization import QuadraticProgram

def qaoa_maxcut(graph, p=2):
    """QAOA for MaxCut problem"""
    # Create problem
    qp = QuadraticProgram()
    for i in range(len(graph.nodes)):
        qp.binary_var(f'x_{i}')
    
    # Objective function
    linear = {}
    quadratic = {}
    for (i, j) in graph.edges:
        quadratic[(i, j)] = 1
        linear[i] = linear.get(i, 0) - 1
        linear[j] = linear.get(j, 0) - 1
    
    # Create QAOA ansatz
    ansatz = QAOAAnsatz(cost_operator=cost_op, p=p)
    
    return ansatz
```

**ML Applications:**
- Feature selection
- Clustering
- Neural network training
- Portfolio optimization

### Variational Quantum Eigensolver (VQE)
**Finding Ground States:**
```python
from qiskit.algorithms import VQE
from qiskit.circuit.library import TwoLocal
from qiskit_nature.problems import ElectronicStructureProblem

def vqe_algorithm(hamiltonian, ansatz=None):
    """VQE for finding minimum eigenvalue"""
    if ansatz is None:
        ansatz = TwoLocal(num_qubits=4, 
                          rotation_blocks='ry', 
                          entanglement_blocks='cz')
    
    # Classical optimizer
    optimizer = COBYLA(maxiter=1000)
    
    # VQE instance
    vqe = VQE(ansatz=ansatz,
              optimizer=optimizer,
              quantum_instance=backend)
    
    result = vqe.compute_minimum_eigenvalue(hamiltonian)
    return result
```

## Quantum Linear Algebra

### HHL Algorithm (Quantum Linear Systems)
**Solving Ax = b:**
```python
from qiskit.circuit.library import HHL

def quantum_linear_solver(matrix_A, vector_b):
    """Harrow-Hassidim-Lloyd algorithm"""
    # Prepare HHL circuit
    hhl = HHL(epsilon=0.01, quantum_instance=backend)
    
    # Solve system
    solution = hhl.solve(matrix_A, vector_b)
    
    return solution
```

**ML Applications:**
- Linear regression
- Support vector machines
- Neural network training
- Recommendation systems

### Quantum Phase Estimation (QPE)
**Eigenvalue Estimation:**
```python
def qpe_circuit(unitary, n_counting_qubits, initial_state=None):
    """Quantum Phase Estimation"""
    n_state_qubits = unitary.num_qubits
    qr_counting = QuantumRegister(n_counting_qubits, 'counting')
    qr_state = QuantumRegister(n_state_qubits, 'state')
    cr = ClassicalRegister(n_counting_qubits, 'c')
    qc = QuantumCircuit(qr_counting, qr_state, cr)
    
    # Initialize counting qubits
    qc.h(qr_counting)
    
    # Initialize state register
    if initial_state:
        qc.append(initial_state, qr_state)
    
    # Controlled unitary operations
    for i in range(n_counting_qubits):
        for _ in range(2**i):
            qc.append(unitary.control(1), 
                     [qr_counting[i]] + list(qr_state))
    
    # Inverse QFT
    qc.append(QFT(n_counting_qubits).inverse(), qr_counting)
    
    # Measure
    qc.measure(qr_counting, cr)
    
    return qc
```

### Quantum Singular Value Decomposition
**Matrix Decomposition:**
- Principal component analysis
- Dimensionality reduction
- Recommender systems
- Data compression

## Variational Quantum Algorithms

### Quantum Neural Networks
**Parametrized Quantum Circuits:**
```python
import pennylane as qml
import numpy as np

def quantum_neural_network(n_qubits, n_layers):
    """Create a QNN"""
    dev = qml.device('default.qubit', wires=n_qubits)
    
    @qml.qnode(dev)
    def circuit(inputs, weights):
        # Encode inputs
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)
        
        # Variational layers
        for l in range(n_layers):
            # Rotation layer
            for i in range(n_qubits):
                qml.RY(weights[l, i, 0], wires=i)
                qml.RZ(weights[l, i, 1], wires=i)
            
            # Entanglement layer
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[n_qubits - 1, 0])
        
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    return circuit
```

### Quantum Autoencoders
**Compression & Denoising:**
```python
def quantum_autoencoder(n_input, n_latent, n_trash):
    """Quantum autoencoder for compression"""
    n_qubits = n_input + n_latent + n_trash
    
    # Encoder
    encoder = QuantumCircuit(n_qubits)
    # ... encoding operations ...
    
    # Decoder
    decoder = encoder.inverse()
    
    # Training circuit
    qc = QuantumCircuit(n_qubits)
    qc.append(encoder, range(n_qubits))
    # Measure trash qubits
    qc.measure(range(n_input, n_input + n_trash), 
               range(n_trash))
    qc.append(decoder, range(n_qubits))
    
    return qc
```

### Quantum GANs
**Generative Models:**
```python
class QuantumGenerator:
    def __init__(self, n_qubits, n_generators):
        self.n_qubits = n_qubits
        self.n_generators = n_generators
        
    def circuit(self, angles):
        qc = QuantumCircuit(self.n_qubits)
        
        # Initial superposition
        qc.h(range(self.n_qubits))
        
        # Parameterized rotations
        for i in range(self.n_qubits):
            qc.ry(angles[i], i)
            qc.rz(angles[i + self.n_qubits], i)
        
        # Entanglement
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        
        return qc

class QuantumDiscriminator:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        
    def circuit(self, data):
        qc = QuantumCircuit(self.n_qubits + 1)
        
        # Encode data
        for i in range(self.n_qubits):
            qc.ry(data[i], i)
        
        # Discriminator network
        # ... variational layers ...
        
        # Measure ancilla for decision
        qc.measure(self.n_qubits, 0)
        
        return qc
```

## Quantum Sampling Algorithms

### Quantum Boltzmann Sampling
**Thermal State Preparation:**
```python
from qiskit.algorithms import QAOA
from qiskit.opflow import I, Z, X

def quantum_boltzmann_machine(hamiltonian, beta):
    """Sample from Gibbs distribution"""
    # Prepare thermal state
    thermal_state = prepare_thermal_state(hamiltonian, beta)
    
    # Sampling circuit
    qc = QuantumCircuit(hamiltonian.num_qubits)
    qc.append(thermal_state, range(hamiltonian.num_qubits))
    qc.measure_all()
    
    return qc
```

### Quantum Markov Chain Monte Carlo
**QMCMC Sampling:**
- Faster mixing times
- Quantum walk advantages
- Applications in optimization
- Bayesian inference

## Hybrid Classical-Quantum Algorithms

### Quantum Kernel Methods
**Feature Maps:**
```python
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
from qiskit_machine_learning.kernels import QuantumKernel

def quantum_kernel_svm(X_train, y_train):
    """Quantum kernel SVM"""
    # Feature map
    feature_map = ZZFeatureMap(
        feature_dimension=X_train.shape[1],
        reps=2,
        entanglement='full'
    )
    
    # Quantum kernel
    quantum_kernel = QuantumKernel(
        feature_map=feature_map,
        quantum_instance=backend
    )
    
    # Classical SVM with quantum kernel
    svm = SVC(kernel=quantum_kernel.evaluate)
    svm.fit(X_train, y_train)
    
    return svm
```

### Quantum Transfer Learning
**Pre-trained Quantum Models:**
```python
def quantum_transfer_learning(pretrained_circuit, new_data):
    """Transfer learning with quantum circuits"""
    # Freeze pretrained layers
    frozen_circuit = pretrained_circuit.copy()
    
    # Add new trainable layers
    n_qubits = frozen_circuit.num_qubits
    trainable = QuantumCircuit(n_qubits)
    
    # New variational layers
    theta = Parameter('θ')
    for i in range(n_qubits):
        trainable.ry(theta, i)
    
    # Combine circuits
    full_circuit = frozen_circuit + trainable
    
    return full_circuit
```

## Implementation Examples

### Complete QAOA Implementation
```python
import networkx as nx
from qiskit import Aer
from qiskit.algorithms.optimizers import COBYLA
from qiskit.algorithms import QAOA
from qiskit_optimization.applications import MaxCut

def solve_maxcut_qaoa(graph, p=3):
    """Complete QAOA for MaxCut"""
    # Create MaxCut instance
    maxcut = MaxCut(graph)
    qp = maxcut.to_quadratic_program()
    
    # Convert to Ising Hamiltonian
    h, offset = qp.to_ising()
    
    # Set up QAOA
    optimizer = COBYLA(maxiter=100)
    backend = Aer.get_backend('qasm_simulator')
    
    qaoa = QAOA(
        optimizer=optimizer,
        reps=p,
        quantum_instance=backend
    )
    
    # Solve
    result = qaoa.compute_minimum_eigenvalue(h)
    
    # Extract solution
    x = maxcut.sample_most_likely(result.eigenstate)
    
    return x, result.eigenvalue.real + offset
```

### Quantum Machine Learning Pipeline
```python
class QuantumMLPipeline:
    def __init__(self, n_qubits, feature_map, ansatz):
        self.n_qubits = n_qubits
        self.feature_map = feature_map
        self.ansatz = ansatz
        self.parameters = None
        
    def encode_data(self, X):
        """Encode classical data to quantum states"""
        circuits = []
        for x in X:
            qc = QuantumCircuit(self.n_qubits)
            qc.append(self.feature_map.bind_parameters(x), 
                     range(self.n_qubits))
            circuits.append(qc)
        return circuits
    
    def train(self, X_train, y_train, epochs=100):
        """Train the quantum model"""
        # Initialize parameters
        self.parameters = np.random.randn(self.ansatz.num_parameters)
        
        for epoch in range(epochs):
            # Forward pass
            predictions = self.predict(X_train)
            
            # Compute loss
            loss = np.mean((predictions - y_train)**2)
            
            # Parameter update (gradient-free for simplicity)
            self.parameters += 0.1 * np.random.randn(*self.parameters.shape)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, X):
        """Make predictions"""
        encoded = self.encode_data(X)
        predictions = []
        
        for circuit in encoded:
            # Add variational ansatz
            circuit.append(
                self.ansatz.bind_parameters(self.parameters),
                range(self.n_qubits)
            )
            
            # Measure
            circuit.measure_all()
            
            # Execute
            result = execute(circuit, backend, shots=1000).result()
            counts = result.get_counts()
            
            # Process results
            prediction = self._process_counts(counts)
            predictions.append(prediction)
        
        return np.array(predictions)
```

## Performance Analysis

### Complexity Comparison
| Algorithm | Classical | Quantum | Speedup |
|-----------|-----------|---------|---------|
| Search (unstructured) | O(N) | O(√N) | Quadratic |
| Factoring | O(exp(n^1/3)) | O(n³) | Exponential |
| Linear Systems | O(n²κ) | O(log(n)κ²) | Exponential* |
| Optimization (QAOA) | NP-hard | Potential advantage | Problem-dependent |

*Under specific conditions

### Resource Requirements
```python
def estimate_resources(algorithm, problem_size):
    """Estimate quantum resources needed"""
    resources = {
        'grover': {
            'qubits': np.log2(problem_size),
            'gates': np.sqrt(problem_size),
            'depth': np.sqrt(problem_size)
        },
        'hhl': {
            'qubits': 3 * np.log2(problem_size) + 10,
            'gates': np.polylog(problem_size),
            'depth': np.polylog(problem_size)
        },
        'vqe': {
            'qubits': problem_size,
            'gates': problem_size ** 2,
            'depth': problem_size
        }
    }
    
    return resources.get(algorithm, {})
```

### Noise Considerations
**Error Mitigation Strategies:**
1. **Zero-Noise Extrapolation**
2. **Probabilistic Error Cancellation**
3. **Symmetry Verification**
4. **Virtual Distillation**

```python
from qiskit.ignis.mitigation import CompleteMeasFitter

def error_mitigation(circuits, backend):
    """Apply measurement error mitigation"""
    # Calibration circuits
    cal_circuits, state_labels = complete_meas_cal(
        qr=circuits[0].qregs[0]
    )
    
    # Execute calibration
    cal_results = execute(cal_circuits, backend).result()
    
    # Build mitigation filter
    meas_fitter = CompleteMeasFitter(cal_results, state_labels)
    meas_filter = meas_fitter.filter
    
    # Execute actual circuits
    results = execute(circuits, backend).result()
    
    # Apply mitigation
    mitigated_results = meas_filter.apply(results)
    
    return mitigated_results
```

## Future Directions

### Fault-Tolerant Algorithms
- Quantum error correction
- Logical qubit operations
- Threshold theorems
- Topological quantum computing

### Novel Algorithm Development
- Quantum reinforcement learning
- Quantum natural language processing
- Quantum computer vision
- Quantum federated learning

### Hardware-Efficient Algorithms
- Native gate decompositions
- Connectivity-aware circuits
- Pulse-level optimization
- Co-design approaches