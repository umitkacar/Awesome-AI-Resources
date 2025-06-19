# Graph Neural Networks (GNNs)

Comprehensive guide to Graph Neural Networks, architectures, applications, and implementation frameworks.

**Last Updated:** 2025-06-19

## Table of Contents
- [Introduction](#introduction)
- [GNN Architectures](#gnn-architectures)
- [Applications](#applications)
- [Frameworks & Libraries](#frameworks--libraries)
- [Datasets & Benchmarks](#datasets--benchmarks)
- [Implementation Guide](#implementation-guide)
- [Advanced Topics](#advanced-topics)
- [Resources](#resources)

## Introduction

Graph Neural Networks extend deep learning to graph-structured data, enabling learning on:
- Social networks
- Molecular structures
- Knowledge graphs
- Transportation networks
- Recommendation systems
- 3D meshes

### Key Concepts
- **Nodes**: Entities in the graph
- **Edges**: Relationships between entities
- **Features**: Node/edge attributes
- **Message Passing**: Information exchange
- **Aggregation**: Combining neighbor information

## GNN Architectures

### Graph Convolutional Networks (GCN)
**[GCN](https://arxiv.org/abs/1609.02907)** - Spectral approach
- 🟢 Simple and effective
- Semi-supervised learning
- Transductive setting

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)
        self.dropout = torch.nn.Dropout(0.5)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
```

### GraphSAGE
**[GraphSAGE](http://snap.stanford.edu/graphsage/)** - Inductive learning
- 🟡 Scalable
- Sampling strategy
- Multiple aggregators

```python
from torch_geometric.nn import SAGEConv

class GraphSAGE(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = SAGEConv(num_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, num_classes)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
```

### Graph Attention Networks (GAT)
**[GAT](https://arxiv.org/abs/1710.10903)** - Attention mechanism
- 🟡 Adaptive weights
- Multi-head attention
- Interpretable

```python
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GATConv(num_features, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 8, num_classes, heads=1, concat=False, dropout=0.6)
    
    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
```

### Graph Isomorphism Network (GIN)
**[GIN](https://arxiv.org/abs/1810.00826)** - Powerful expressiveness
- 🔴 Theoretical foundation
- WL-test equivalent
- Provably powerful

### Message Passing Neural Networks (MPNN)
**[MPNN](https://arxiv.org/abs/1704.01212)** - General framework
- Unified view
- Customizable functions
- Molecular property prediction

## Applications

### Social Network Analysis
**Node Classification**
- User profiling
- Community detection
- Influence prediction

```python
# Example: Predicting user interests
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# Build model for node classification
model = GCN(
    num_features=dataset.num_features,
    num_classes=dataset.num_classes
)
```

### Molecular Property Prediction
**[MoleculeNet](http://moleculenet.ai/)** - Drug discovery
- SMILES to graphs
- Quantum properties
- Toxicity prediction

```python
from rdkit import Chem
from torch_geometric.utils import from_smiles

# Convert SMILES to graph
smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'  # Aspirin
data = from_smiles(smiles)

# Predict properties
model = MPNN(
    node_features=data.x.shape[1],
    edge_features=data.edge_attr.shape[1],
    output_dim=1  # e.g., solubility
)
```

### Knowledge Graph Reasoning
**Link Prediction**
- Missing fact prediction
- Entity alignment
- Relation extraction

```python
from torch_geometric.nn import RGCNConv

class RGCN(torch.nn.Module):
    def __init__(self, num_entities, num_relations, hidden_dim):
        super().__init__()
        self.entity_embedding = torch.nn.Embedding(num_entities, hidden_dim)
        self.conv1 = RGCNConv(hidden_dim, hidden_dim, num_relations)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations)
    
    def forward(self, edge_index, edge_type):
        x = self.entity_embedding.weight
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = self.conv2(x, edge_index, edge_type)
        return x
```

### Recommendation Systems
**[PinSage](https://arxiv.org/abs/1806.01973)** - Pinterest's GNN
- Billion-scale graphs
- Random walk sampling
- Production system

### Computer Vision
**3D Point Clouds**
- Object classification
- Scene segmentation
- Shape completion

```python
from torch_geometric.nn import PointConv

class PointNet(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = PointConv(local_nn=MLP([3, 64, 128]))
        self.conv2 = PointConv(local_nn=MLP([128, 256, 512]))
        self.classifier = torch.nn.Linear(512, num_classes)
    
    def forward(self, pos, batch):
        # Point cloud processing
        x = self.conv1(x=None, pos=pos, edge_index=radius_graph(pos, r=0.2, batch=batch))
        x = self.conv2(x=x, pos=pos, edge_index=radius_graph(pos, r=0.4, batch=batch))
        x = global_max_pool(x, batch)
        return self.classifier(x)
```

## Frameworks & Libraries

### PyTorch Geometric
**[PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)** - Most popular
- 🆓 Open source
- 🟢 Easy to use
- Extensive models
- GPU acceleration

```bash
# Installation
pip install torch-geometric
```

Key features:
- 60+ GNN layers
- 40+ benchmark datasets
- Mini-batch loaders
- Multi-GPU support

### DGL (Deep Graph Library)
**[DGL](https://www.dgl.ai/)** - Multi-backend
- 🆓 Amazon's solution
- PyTorch/TensorFlow/MXNet
- Scalable
- Production ready

```python
import dgl
import dgl.nn as dglnn

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_feats, hidden_size)
        self.conv2 = dglnn.GraphConv(hidden_size, num_classes)
    
    def forward(self, g, features):
        h = F.relu(self.conv1(g, features))
        h = self.conv2(g, h)
        return h
```

### Spektral
**[Spektral](https://graphneural.network/)** - Keras/TensorFlow
- 🆓 Open source
- 🟢 Keras-like API
- Graph classification focus
- Good documentation

### Jraph
**[Jraph](https://github.com/deepmind/jraph)** - JAX-based
- 🆓 DeepMind's library
- Functional approach
- JIT compilation
- Research focused

### NetworkX Integration
```python
import networkx as nx
from torch_geometric.utils import from_networkx

# Create NetworkX graph
G = nx.karate_club_graph()

# Convert to PyTorch Geometric
data = from_networkx(G)

# Add node features
data.x = torch.eye(G.number_of_nodes())
```

## Datasets & Benchmarks

### Citation Networks
| Dataset | Nodes | Edges | Classes | Features |
|---------|-------|-------|---------|----------|
| Cora | 2,708 | 5,429 | 7 | 1,433 |
| CiteSeer | 3,327 | 4,732 | 6 | 3,703 |
| PubMed | 19,717 | 44,338 | 3 | 500 |

### Social Networks
- **[Facebook](https://snap.stanford.edu/data/)** - Page-page networks
- **[Reddit](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.Reddit)** - Community detection
- **[Twitter](https://snap.stanford.edu/data/ego-Twitter.html)** - Ego networks

### Molecular Datasets
- **[QM9](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.QM9)** - 134k molecules
- **[ZINC](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.ZINC)** - 250k molecules
- **[MoleculeNet](http://moleculenet.ai/)** - Multiple benchmarks

### Benchmark Suites
- **[Open Graph Benchmark (OGB)](https://ogb.stanford.edu/)** - Large-scale benchmarks
- **[TUDataset](https://chrsmrrs.github.io/datasets/)** - Graph classification
- **[GraphGym](https://github.com/snap-stanford/GraphGym)** - Modular framework

## Implementation Guide

### Data Preparation
```python
import torch
from torch_geometric.data import Data

# Create graph data
edge_index = torch.tensor([
    [0, 1, 1, 2],
    [1, 0, 2, 1]
], dtype=torch.long)

x = torch.tensor([
    [-1], [0], [1]
], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)
```

### Training Loop
```python
def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, data):
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum())
    return acc
```

### Mini-batch Training
```python
from torch_geometric.loader import NeighborLoader

# Create mini-batch loader
train_loader = NeighborLoader(
    data,
    num_neighbors=[10, 10],  # Sample 10 neighbors for 2 hops
    batch_size=128,
    input_nodes=data.train_mask,
    shuffle=True
)

# Training with batches
for batch in train_loader:
    out = model(batch.x, batch.edge_index)
    loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size])
```

## Advanced Topics

### Graph Pooling
```python
from torch_geometric.nn import global_mean_pool, TopKPooling

class GraphClassifier(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, 64)
        self.pool1 = TopKPooling(64, ratio=0.8)
        self.conv2 = GCNConv(64, 64)
        self.pool2 = TopKPooling(64, ratio=0.8)
        self.classifier = torch.nn.Linear(64, num_classes)
    
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        
        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        
        x = global_mean_pool(x, batch)
        return self.classifier(x)
```

### Temporal Graphs
**Dynamic GNNs**
- Evolving networks
- Temporal dependencies
- Event prediction

```python
class TemporalGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim):
        super().__init__()
        self.gnn = GCNConv(num_features, hidden_dim)
        self.rnn = torch.nn.GRU(hidden_dim, hidden_dim)
    
    def forward(self, x_seq, edge_index_seq):
        h = None
        outputs = []
        
        for t in range(len(x_seq)):
            x = self.gnn(x_seq[t], edge_index_seq[t])
            x, h = self.rnn(x.unsqueeze(0), h)
            outputs.append(x.squeeze(0))
        
        return torch.stack(outputs)
```

### Explainability
**[GNNExplainer](https://arxiv.org/abs/1903.03894)** - Interpretable GNNs
```python
from torch_geometric.explain import GNNExplainer

explainer = GNNExplainer(model, epochs=200)
node_idx = 10
node_feat_mask, edge_mask = explainer.explain_node(
    node_idx, 
    data.x, 
    data.edge_index
)
```

### Scalability Techniques
1. **Sampling Methods**
   - Neighbor sampling
   - Layer-wise sampling
   - Importance sampling

2. **Graph Partitioning**
   - METIS partitioning
   - Cluster-GCN
   - GraphSAINT

3. **Distributed Training**
   - DGL distributed
   - PyG distributed
   - Horovod integration

## Best Practices

### Model Selection
1. **Small graphs (<10k nodes)**: Full-batch GCN/GAT
2. **Medium graphs (<1M nodes)**: GraphSAGE with sampling
3. **Large graphs (>1M nodes)**: Cluster-GCN, GraphSAINT
4. **Dynamic graphs**: Temporal GNNs, EvolveGCN

### Hyperparameter Tuning
```python
# Common hyperparameters
hyperparams = {
    'hidden_dim': [64, 128, 256],
    'num_layers': [2, 3, 4],
    'dropout': [0.0, 0.2, 0.5],
    'learning_rate': [0.01, 0.005, 0.001],
    'weight_decay': [0, 5e-4, 1e-3]
}
```

### Performance Optimization
- Use sparse operations
- Precompute node features
- Cache aggregation results
- GPU memory management

## Resources

### Courses & Tutorials
- [CS224W: Machine Learning with Graphs](http://web.stanford.edu/class/cs224w/) - Stanford
- [Geometric Deep Learning](https://geometricdeeplearning.com/) - Course & book
- [PyG Tutorials](https://pytorch-geometric.readthedocs.io/en/latest/notes/colabs.html) - Official tutorials

### Papers
- [Graph Neural Networks: A Review](https://arxiv.org/abs/1901.00596)
- [Benchmarking GNNs](https://arxiv.org/abs/2003.00982)
- [Design Space for GNNs](https://arxiv.org/abs/2011.08843)

### Tools & Visualization
- [GraphGym](https://github.com/snap-stanford/GraphGym) - Design GNN models
- [PyTorch Geometric Temporal](https://pytorch-geometric-temporal.readthedocs.io/) - Temporal graphs
- [NetworkX](https://networkx.org/) - Graph algorithms
- [Gephi](https://gephi.org/) - Graph visualization

### Communities
- [r/GraphNeuralNetworks](https://reddit.com/r/graphneuralnetworks) - Reddit
- [Graph ML Slack](https://join.slack.com/t/graphml/shared_invite/zt-1hk7q7e7j-FmPttFgPEHabGsOmKvCyvg) - Community