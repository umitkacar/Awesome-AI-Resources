# Quantum Machine Learning Frameworks & Tools

Comprehensive guide to frameworks, libraries, and tools for quantum machine learning development.

**Last Updated:** 2025-06-19

## Table of Contents
- [Major QML Frameworks](#major-qml-frameworks)
- [Quantum Circuit Libraries](#quantum-circuit-libraries)
- [Quantum Simulators](#quantum-simulators)
- [Development Tools](#development-tools)
- [Cloud Platforms](#cloud-platforms)
- [Visualization Tools](#visualization-tools)
- [Benchmarking & Testing](#benchmarking--testing)
- [Integration Examples](#integration-examples)

## Major QML Frameworks

### PennyLane
**[PennyLane](https://pennylane.ai/)** - Cross-platform Python library for quantum ML
- 🆓 Open source (Apache 2.0)
- 🟢 Most beginner-friendly
- Device agnostic
- AutoDiff support
- Extensive documentation

**Key Features:**
```python
import pennylane as qml
import numpy as np

# Create device
dev = qml.device('default.qubit', wires=4)

# Quantum node decorator
@qml.qnode(dev)
def quantum_circuit(params, x):
    # Data encoding
    qml.AngleEmbedding(x, wires=range(4))
    
    # Trainable layers
    qml.StronglyEntanglingLayers(params, wires=range(4))
    
    # Measurement
    return qml.expval(qml.PauliZ(0))

# Automatic differentiation
grad_fn = qml.grad(quantum_circuit)
```

**Supported Backends:**
- IBM Qiskit
- Google Cirq
- Rigetti Forest
- IonQ
- AWS Braket
- Xanadu Strawberry Fields

### TensorFlow Quantum
**[TensorFlow Quantum (TFQ)](https://www.tensorflow.org/quantum)** - Google's QML framework
- 🆓 Open source
- 🟡 Intermediate level
- TensorFlow integration
- High performance
- Research focused

**Example Implementation:**
```python
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq

# Create quantum circuit
def create_quantum_model():
    # Define qubits
    qubits = cirq.GridQubit.rect(1, 4)
    
    # Input circuit
    input_circuit = cirq.Circuit()
    
    # Model circuit with parameters
    model_circuit = cirq.Circuit()
    for i, qubit in enumerate(qubits):
        model_circuit += cirq.ry(sympy.Symbol(f'theta_{i}'))(qubit)
    
    # Readout operators
    readouts = [cirq.Z(q) for q in qubits]
    
    # Build Keras model
    model = tf.keras.Sequential([
        tfq.layers.PQC(model_circuit, readouts)
    ])
    
    return model
```

### Qiskit Machine Learning
**[Qiskit ML](https://qiskit.org/documentation/machine-learning/)** - IBM's quantum ML module
- 🆓 Open source
- 🟡 Intermediate
- Real hardware access
- Comprehensive algorithms
- Active development

**Core Components:**
```python
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.algorithms import VQC, VQR
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap

# Quantum Neural Network
feature_map = ZZFeatureMap(feature_dimension=4)
ansatz = RealAmplitudes(num_qubits=4, reps=3)

qnn = EstimatorQNN(
    circuit=feature_map.compose(ansatz),
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters
)

# Variational Quantum Classifier
vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=COBYLA(maxiter=100)
)
```

### Amazon Braket SDK
**[Amazon Braket](https://github.com/aws/amazon-braket-sdk-python)** - AWS quantum computing
- 💰 Pay-per-use
- Multiple backends
- Managed notebooks
- Hybrid algorithms
- Production ready

**Usage Example:**
```python
from braket.circuits import Circuit
from braket.devices import LocalSimulator
from braket.aws import AwsDevice

# Create circuit
circuit = Circuit().h(0).cnot(0, 1)

# Local simulation
device = LocalSimulator()
result = device.run(circuit, shots=1000).result()

# Run on quantum hardware
device = AwsDevice("arn:aws:braket::device/qpu/ionq/ionQdevice")
task = device.run(circuit, shots=100)
```

## Quantum Circuit Libraries

### Cirq
**[Cirq](https://quantumai.google/cirq)** - Google's quantum circuits framework
- 🆓 Open source
- Low-level control
- NISQ focused
- Noise modeling
- Hardware optimization

```python
import cirq

# Create circuit
q0, q1 = cirq.LineQubit.range(2)
circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.CNOT(q0, q1),
    cirq.measure(q0, q1, key='result')
)

# Simulate
simulator = cirq.Simulator()
result = simulator.run(circuit, repetitions=100)
```

### PyQuil
**[PyQuil](https://pyquil-docs.rigetti.com/)** - Rigetti's quantum programming
- Forest SDK integration
- Quil language
- Quantum/classical hybrid
- Compiler optimizations

### ProjectQ
**[ProjectQ](https://projectq.ch/)** - Open source quantum computing
- High-level abstractions
- Automatic compilation
- Hardware backends
- Emulator included

### Strawberry Fields
**[Strawberry Fields](https://strawberryfields.ai/)** - Photonic quantum computing
- 🆓 Open source
- Continuous variables
- Gaussian states
- Photonic devices
- GBS algorithms

```python
import strawberryfields as sf
from strawberryfields.ops import *

# Photonic circuit
prog = sf.Program(2)
with prog.context as q:
    # Gaussian operations
    Sgate(0.5) | q[0]
    BSgate(0.5, 0.1) | (q[0], q[1])
    MeasureFock() | q

# Run simulation
eng = sf.Engine("gaussian")
result = eng.run(prog)
```

## Quantum Simulators

### State Vector Simulators
**Qiskit Aer:**
```python
from qiskit import Aer
from qiskit.providers.aer import AerSimulator

# GPU-accelerated simulator
simulator = AerSimulator(device='GPU')

# Statevector simulation
backend = Aer.get_backend('statevector_simulator')
```

**Limits:**
- ~30 qubits (32GB RAM)
- Exponential scaling
- Full state access
- Exact results

### Tensor Network Simulators
**[TensorNetwork](https://github.com/google/TensorNetwork)** - Google's TN library
```python
import tensornetwork as tn

# Create tensor network
a = tn.Node(np.random.randn(2, 3, 4))
b = tn.Node(np.random.randn(4, 5, 2))

# Connect edges
edge = a[2] ^ b[0]
tn.contract(edge)
```

**Advantages:**
- 100+ qubits possible
- Low entanglement circuits
- Memory efficient
- Approximate methods

### Noise Simulators
**Qiskit Noise Models:**
```python
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error

# Build noise model
noise_model = NoiseModel()

# Single-qubit error
error_1 = depolarizing_error(0.001, 1)
noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])

# Two-qubit error
error_2 = depolarizing_error(0.01, 2)
noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
```

## Development Tools

### Quantum Development Environments
**[IBM Quantum Lab](https://quantum-computing.ibm.com/lab)** - Cloud-based Jupyter
- Pre-installed libraries
- Hardware access
- Tutorials included
- Free tier available

**[Microsoft Quantum Development Kit](https://azure.microsoft.com/en-us/products/quantum-development-kit/)** - Q# language
- Visual Studio integration
- Azure Quantum access
- Resource estimation
- Debugging tools

### Compiler & Optimization Tools
**[t|ket⟩](https://github.com/CQCL/tket)** - Cambridge Quantum Computing
```python
from pytket import Circuit
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit

# Optimize Qiskit circuit
qiskit_circuit = create_circuit()
tk_circuit = qiskit_to_tk(qiskit_circuit)

# Apply optimizations
from pytket.passes import FullPeepholeOptimise
FullPeepholeOptimise().apply(tk_circuit)

# Convert back
optimized = tk_to_qiskit(tk_circuit)
```

**[Mitiq](https://mitiq.readthedocs.io/)** - Error mitigation
```python
from mitiq import zne, execute_with_zne

def execute(circuit, noise_level=0.01):
    """Noisy execution function"""
    return noisy_backend.run(circuit).result()

# Zero-noise extrapolation
ideal_result = execute_with_zne(
    circuit, 
    execute,
    scale_noise=zne.scaling.fold_global
)
```

### Package Managers
**pip packages:**
```bash
# Core frameworks
pip install pennylane qiskit tensorflow-quantum

# Additional tools
pip install amazon-braket-sdk cirq projectq

# Visualization
pip install qiskit[visualization] pylatexenc
```

**conda environments:**
```bash
# Create QML environment
conda create -n qml python=3.9
conda activate qml

# Install frameworks
conda install -c conda-forge pennylane
conda install -c conda-forge qiskit
```

## Cloud Platforms

### IBM Quantum Network
**Features:**
- 20+ quantum systems
- Up to 127 qubits
- Free tier (10 min/month)
- Qiskit Runtime
- Fair queue system

**Access Pattern:**
```python
from qiskit import IBMQ

# Load account
IBMQ.load_account()

# Get backend
provider = IBMQ.get_provider(hub='ibm-q')
backend = provider.get_backend('ibmq_montreal')

# Check properties
print(backend.configuration())
print(backend.properties())
```

### AWS Braket
**Available Devices:**
- IonQ (trapped ion)
- Rigetti (superconducting)
- Oxford Quantum Circuits
- Quantum simulators

**Pricing:**
- Per-shot pricing
- Per-minute pricing
- Simulator: $0.075/min
- Hardware: $0.30-$0.80/shot

### Azure Quantum
**Partners:**
- IonQ
- Honeywell
- QCI
- 1QBit

**Integration:**
```python
from azure.quantum import Workspace
from azure.quantum.qiskit import AzureQuantumProvider

# Connect to workspace
workspace = Workspace(
    subscription_id="...",
    resource_group="...",
    name="...",
    location="..."
)

# Get provider
provider = AzureQuantumProvider(workspace)
backend = provider.get_backend("ionq.simulator")
```

### Google Quantum AI
**Access:**
- Limited availability
- Research partnerships
- Cirq integration
- Sycamore processor

## Visualization Tools

### Circuit Visualization
**Qiskit Visualization:**
```python
from qiskit.visualization import circuit_drawer, plot_histogram

# Draw circuit
circuit_drawer(circuit, output='mpl', style='iqx')

# Plot results
plot_histogram(counts)
```

**Cirq Visualization:**
```python
# Text representation
print(circuit)

# SVG output
cirq.contrib.svg.SVGCircuit(circuit)
```

### State Visualization
**Bloch Sphere:**
```python
from qiskit.visualization import plot_bloch_multivector

# Visualize quantum state
state = qi.Statevector.from_instruction(circuit)
plot_bloch_multivector(state)
```

**Density Matrix:**
```python
from qiskit.visualization import plot_state_city

# City plot
plot_state_city(density_matrix)
```

### Web-Based Tools
**[Quirk](https://algassert.com/quirk)** - Quantum circuit simulator
- Browser-based
- Drag-and-drop
- Real-time simulation
- Export circuits

**[IBM Quantum Composer](https://quantum-computing.ibm.com/composer)** - Visual circuit builder
- Hardware access
- Built-in examples
- OpenQASM export

## Benchmarking & Testing

### Quantum Benchmarking
**[QASMBench](https://github.com/pnnl/QASMBench)** - Benchmark suite
```python
# Benchmark categories
benchmarks = {
    'small': '2-5 qubits',
    'medium': '6-15 qubits', 
    'large': '16+ qubits'
}
```

**[SupermarQ](https://github.com/SupermarQ/SupermarQ)** - Application benchmarks
- Feature-based metrics
- Hardware comparison
- Realistic workloads

### Unit Testing
**PennyLane Testing:**
```python
import pytest
import pennylane as qml

def test_quantum_circuit():
    dev = qml.device('default.qubit', wires=2)
    
    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(0)
        qml.CNOT(wires=[0, 1])
        return qml.state()
    
    state = circuit()
    expected = np.array([1, 0, 0, 1]) / np.sqrt(2)
    assert np.allclose(state, expected)
```

### Performance Profiling
**Qiskit Profiling:**
```python
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Depth, CountOps

# Analyze circuit
pm = PassManager([Depth(), CountOps()])
pm.run(circuit)

print(f"Depth: {circuit.depth()}")
print(f"Gate count: {circuit.count_ops()}")
```

## Integration Examples

### Hybrid Classical-Quantum
**PyTorch + PennyLane:**
```python
import torch
import pennylane as qml

# Hybrid model
class HybridModel(torch.nn.Module):
    def __init__(self, n_qubits, n_classes):
        super().__init__()
        self.classical = torch.nn.Linear(10, n_qubits)
        self.quantum = qml.qnode(dev, interface='torch')(quantum_circuit)
        self.final = torch.nn.Linear(n_qubits, n_classes)
    
    def forward(self, x):
        x = self.classical(x)
        x = self.quantum(x)
        x = self.final(x)
        return x
```

### Multi-Framework Pipeline
```python
# Qiskit → PennyLane → TensorFlow
from pennylane_qiskit import load

# Load Qiskit circuit
qiskit_circuit = QuantumCircuit(4)
# ... build circuit ...

# Convert to PennyLane
pl_circuit = load(qiskit_circuit)

# Use in TensorFlow
import tensorflow as tf

@tf.function
def hybrid_computation(x):
    # Classical preprocessing
    x = tf.nn.relu(x)
    
    # Quantum processing
    quantum_out = pl_circuit(x)
    
    # Classical postprocessing
    return tf.nn.softmax(quantum_out)
```

### Production Deployment
**Flask API Example:**
```python
from flask import Flask, request, jsonify
import pennylane as qml

app = Flask(__name__)

# Initialize quantum device
dev = qml.device('default.qubit', wires=4)

@qml.qnode(dev)
def quantum_model(inputs):
    # ... quantum circuit ...
    return qml.expval(qml.PauliZ(0))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    result = quantum_model(data)
    return jsonify({'prediction': float(result)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## Best Practices

### Framework Selection
1. **PennyLane**: Best for ML researchers, AutoDiff needed
2. **Qiskit**: IBM hardware access, comprehensive tooling
3. **TFQ**: TensorFlow ecosystem, Google research
4. **Cirq**: Low-level control, custom gates
5. **Braket**: Production deployment, multi-vendor

### Development Workflow
```bash
# 1. Prototype locally
python prototype.py

# 2. Test on simulator
python -m pytest tests/

# 3. Optimize circuit
python optimize_circuit.py

# 4. Run on hardware
python run_hardware.py

# 5. Analyze results
python analyze_results.py
```

### Common Pitfalls
- Not checking backend capabilities
- Ignoring transpilation costs
- Assuming simulator = hardware
- Poor parameter initialization
- Inadequate error mitigation

## Resources & Community

### Documentation
- [PennyLane Docs](https://docs.pennylane.ai/)
- [Qiskit Textbook](https://qiskit.org/textbook/)
- [TFQ Tutorials](https://www.tensorflow.org/quantum/tutorials)
- [Cirq Documentation](https://quantumai.google/cirq/docs)

### Community Forums
- [Qiskit Slack](https://qiskit.slack.com/)
- [PennyLane Forum](https://discuss.pennylane.ai/)
- [Quantum Computing SE](https://quantumcomputing.stackexchange.com/)
- [Reddit r/QuantumComputing](https://reddit.com/r/QuantumComputing)

### Contributing
Most frameworks welcome contributions:
```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/pennylane
cd pennylane

# Install in dev mode
pip install -e .

# Run tests
pytest tests/

# Submit PR
git push origin feature-branch
```