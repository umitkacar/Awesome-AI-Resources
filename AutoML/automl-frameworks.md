# AutoML Frameworks & Tools

Comprehensive guide to Automated Machine Learning (AutoML) platforms, frameworks, and tools that democratize AI.

**Last Updated:** 2025-06-19

## Table of Contents
- [What is AutoML?](#what-is-automl)
- [Open Source Frameworks](#open-source-frameworks)
- [Commercial Platforms](#commercial-platforms)
- [Neural Architecture Search](#neural-architecture-search)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Feature Engineering](#feature-engineering)
- [Comparison & Benchmarks](#comparison--benchmarks)
- [Getting Started](#getting-started)

## What is AutoML?

AutoML automates the end-to-end process of applying machine learning to real-world problems, including:
- Data preprocessing
- Feature engineering
- Model selection
- Hyperparameter tuning
- Model deployment

## Open Source Frameworks

### AutoGluon
**[AutoGluon](https://auto.gluon.ai/)** - Amazon's AutoML toolkit
- 🆓 Open source
- 🟢 Beginner friendly
- Multi-modal support
- State-of-the-art performance

```python
from autogluon.tabular import TabularPredictor

# Train with just 3 lines
predictor = TabularPredictor(label='target')
predictor.fit(train_data)
predictions = predictor.predict(test_data)
```

**Key Features:**
- Automatic stacking/ensembling
- GPU support
- Time series forecasting
- Object detection
- Text & image classification

### H2O AutoML
**[H2O AutoML](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)** - Scalable AutoML
- 🆓 Open source
- Distributed computing
- Explainability built-in
- Production ready

```python
import h2o
from h2o.automl import H2OAutoML

h2o.init()
aml = H2OAutoML(max_models=20, seed=1)
aml.train(x=features, y=target, training_frame=train)
```

### Auto-sklearn
**[Auto-sklearn](https://automl.github.io/auto-sklearn/)** - Automated scikit-learn
- 🆓 Open source
- 🟡 Intermediate
- Bayesian optimization
- Meta-learning

```python
import autosklearn.classification

automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=120,
    per_run_time_limit=30
)
automl.fit(X_train, y_train)
```

### FLAML
**[FLAML](https://microsoft.github.io/FLAML/)** - Fast and Lightweight AutoML
- 🆓 Microsoft's solution
- Cost-effective
- Custom learners
- Zero-shot AutoML

```python
from flaml import AutoML

automl = AutoML()
automl.fit(X_train, y_train, task="classification", time_budget=60)
```

### PyCaret
**[PyCaret](https://pycaret.org/)** - Low-code ML library
- 🆓 Open source
- 🟢 Very beginner friendly
- Beautiful visualizations
- 100+ algorithms

```python
from pycaret.classification import *

# Setup
clf1 = setup(data, target='target')

# Compare models
best_model = compare_models()

# Create model
model = create_model('rf')

# Tune hyperparameters
tuned_model = tune_model(model)

# Make predictions
predictions = predict_model(tuned_model, data=test)
```

### TPOT
**[TPOT](http://epistasislab.github.io/tpot/)** - Tree-based Pipeline Optimization
- 🆓 Open source
- Genetic programming
- Scikit-learn pipelines
- Export to Python code

```python
from tpot import TPOTClassifier

tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2)
tpot.fit(X_train, y_train)
tpot.export('best_pipeline.py')
```

## Commercial Platforms

### Google Cloud AutoML
**[Google Cloud AutoML](https://cloud.google.com/automl)** - Enterprise AutoML
- 💰 Pay-per-use
- No-code/low-code
- Multiple modalities
- Vertex AI integration

**Products:**
- AutoML Tables
- AutoML Vision
- AutoML Natural Language
- AutoML Translation
- AutoML Video Intelligence

### Azure AutoML
**[Azure Machine Learning AutoML](https://azure.microsoft.com/en-us/services/machine-learning/automatedml/)** - Microsoft's AutoML
- 💰 Commercial
- Enterprise features
- MLOps integration
- Responsible AI

```python
from azureml.train.automl import AutoMLConfig

automl_config = AutoMLConfig(
    task='classification',
    primary_metric='accuracy',
    training_data=train_data,
    label_column_name='target',
    n_cross_validations=5
)
```

### Amazon SageMaker Autopilot
**[SageMaker Autopilot](https://aws.amazon.com/sagemaker/autopilot/)** - AWS AutoML
- 💰 Pay-per-use
- Full visibility
- Model explainability
- One-click deployment

### DataRobot
**[DataRobot](https://www.datarobot.com/)** - Enterprise AI platform
- 💰 Premium
- End-to-end platform
- MLOps capabilities
- Business user friendly

### H2O Driverless AI
**[H2O Driverless AI](https://www.h2o.ai/products/h2o-driverless-ai/)** - Commercial AutoML
- 💰 Enterprise
- Automatic visualization
- Time series
- NLP & computer vision

## Neural Architecture Search

### AutoKeras
**[AutoKeras](https://autokeras.com/)** - AutoML for deep learning
- 🆓 Open source
- Keras/TensorFlow based
- Image, text, structured data
- Customizable

```python
import autokeras as ak

# Image classifier
clf = ak.ImageClassifier(max_trials=10)
clf.fit(x_train, y_train)

# Neural architecture search
model = clf.export_model()
```

### NAS Frameworks
**[NNI (Neural Network Intelligence)](https://github.com/microsoft/nni)** - Microsoft's AutoML toolkit
- Feature engineering
- Neural architecture search
- Hyperparameter tuning
- Model compression

**[ENAS](https://github.com/melodyguan/enas)** - Efficient Neural Architecture Search
- Parameter sharing
- Faster search
- CIFAR-10: 2.89% error

**[DARTS](https://github.com/quark0/darts)** - Differentiable Architecture Search
- Gradient-based
- Memory efficient
- CNN & RNN search

## Hyperparameter Optimization

### Optuna
**[Optuna](https://optuna.org/)** - Hyperparameter optimization framework
- 🆓 Open source
- Define-by-run API
- Pruning algorithms
- Visualization

```python
import optuna

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2

study = optuna.create_study()
study.optimize(objective, n_trials=100)
```

### Ray Tune
**[Ray Tune](https://docs.ray.io/en/latest/tune/index.html)** - Scalable hyperparameter tuning
- Distributed execution
- Multiple algorithms
- Early stopping
- Population-based training

### Hyperopt
**[Hyperopt](http://hyperopt.github.io/hyperopt/)** - Bayesian optimization
- Tree-structured Parzen Estimator
- Random search
- Adaptive TPE

## Feature Engineering

### Featuretools
**[Featuretools](https://www.featuretools.com/)** - Automated feature engineering
- 🆓 Open source
- Deep feature synthesis
- Time-aware features
- Custom primitives

```python
import featuretools as ft

# Automated feature engineering
feature_matrix, features = ft.dfs(
    entityset=es,
    target_entity="customers",
    max_depth=2
)
```

### TSFresh
**[TSFresh](https://tsfresh.readthedocs.io/)** - Time series feature extraction
- 750+ features
- Automatic selection
- Scalable extraction
- Statistical tests

### AutoFeat
**[AutoFeat](https://github.com/cod3licious/autofeat)** - Automatic feature engineering
- Linear models
- Feature selection
- Non-linear features

## Comparison & Benchmarks

### Performance Comparison

| Framework | Tabular | Images | Text | Time Series | Ease of Use |
|-----------|---------|--------|------|-------------|-------------|
| AutoGluon | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| H2O AutoML | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Auto-sklearn | ⭐⭐⭐⭐ | ❌ | ❌ | ❌ | ⭐⭐⭐ |
| PyCaret | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| FLAML | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |

### Benchmark Datasets
- **OpenML AutoML Benchmark** - Standard evaluation
- **AMLB** - AutoML Benchmark
- **Kaggle Competitions** - Real-world performance

## Getting Started

### Quick Start Template
```python
# 1. Choose your framework
from autogluon.tabular import TabularPredictor
# or
from pycaret.classification import *
# or
import h2o.automl

# 2. Load your data
import pandas as pd
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 3. Run AutoML
# AutoGluon example
predictor = TabularPredictor(label='target')
predictor.fit(train_data, time_limit=300)

# 4. Make predictions
predictions = predictor.predict(test_data)

# 5. Evaluate
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true, predictions)
```

### Best Practices

1. **Data Quality First**
   - Clean data > fancy algorithms
   - Handle missing values
   - Remove outliers
   - Feature engineering still helps

2. **Resource Management**
   - Set time/resource limits
   - Use early stopping
   - Monitor memory usage
   - Consider cloud options

3. **Validation Strategy**
   - Proper train/test split
   - Cross-validation
   - Time series: temporal split
   - Stratification for imbalanced

4. **Interpretability**
   - SHAP values
   - Feature importance
   - Model documentation
   - Business alignment

### When to Use AutoML

✅ **Good Use Cases:**
- Proof of concepts
- Baseline models
- Non-ML experts
- Time constraints
- Multiple datasets

❌ **Not Recommended:**
- Highly specialized domains
- Extreme performance needs
- Limited computational resources
- Need full control
- Regulatory requirements

## Advanced Topics

### Ensemble Strategies
```python
# Multi-layer stacking
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(
    label='target',
    eval_metric='roc_auc'
).fit(
    train_data,
    num_bag_folds=10,
    num_bag_sets=20,
    num_stack_levels=3
)
```

### Custom Metrics
```python
def custom_metric(y_true, y_pred):
    # Business-specific metric
    return business_value_score

predictor.fit(
    train_data,
    metric=custom_metric
)
```

### Production Deployment
```python
# Save model
predictor.save('model.pkl')

# Deploy to cloud
predictor.deploy(
    cloud='aws',
    instance_type='ml.m5.large',
    endpoint_name='automl-endpoint'
)
```

## Resources & Learning

### Tutorials
- [AutoML Course](https://www.automl.org/automl-course/) - Free online course
- [Google's AutoML Guide](https://cloud.google.com/automl/docs)
- [Papers with Code - AutoML](https://paperswithcode.com/task/automl)

### Research Papers
- [AutoML: A Survey of the State-of-the-Art](https://arxiv.org/abs/1908.00709)
- [Efficient and Robust AutoML](https://arxiv.org/abs/1908.06176)
- [Neural Architecture Search: A Survey](https://arxiv.org/abs/1808.05377)

### Community
- [AutoML.org](https://www.automl.org/) - Research community
- [r/AutoML](https://reddit.com/r/automl) - Reddit community
- [AutoML Slack](https://automl.slack.com) - Discussion channel