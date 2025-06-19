# 🔄 AI Data Preprocessing Techniques

**Last Updated:** 2025-06-19

## Overview
Comprehensive guide to data preprocessing techniques for AI/ML projects, covering data cleaning, transformation, feature engineering, and optimization strategies.

## 🎯 Why Preprocessing Matters

### Impact on Model Performance
- **Quality**: "Garbage in, garbage out" - clean data is crucial
- **Features**: Well-engineered features can boost accuracy by 10-30%
- **Training Speed**: Proper scaling can speed up convergence 5-10x
- **Generalization**: Good preprocessing improves model robustness
- **Memory**: Efficient encoding reduces memory usage by 50-90%

## 🧹 Data Cleaning

### Missing Value Handling
```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

class MissingValueHandler:
    def __init__(self, strategy='auto'):
        self.strategy = strategy
        self.imputers = {}
        
    def fit_transform(self, df):
        df_processed = df.copy()
        
        for column in df.columns:
            if df[column].isnull().sum() > 0:
                missing_ratio = df[column].isnull().sum() / len(df)
                
                if missing_ratio > 0.5:
                    # Drop column if too many missing values
                    df_processed = df_processed.drop(column, axis=1)
                    print(f"Dropped {column}: {missing_ratio:.2%} missing")
                    
                elif df[column].dtype in ['object', 'category']:
                    # Categorical: mode or create 'missing' category
                    if missing_ratio < 0.1:
                        mode = df[column].mode()[0]
                        df_processed[column].fillna(mode, inplace=True)
                    else:
                        df_processed[column].fillna('missing', inplace=True)
                        
                else:
                    # Numerical: choose strategy based on distribution
                    if self.strategy == 'auto':
                        strategy = self._choose_strategy(df[column])
                    else:
                        strategy = self.strategy
                    
                    if strategy == 'knn':
                        imputer = KNNImputer(n_neighbors=5)
                    elif strategy == 'iterative':
                        imputer = IterativeImputer(random_state=42)
                    else:
                        imputer = SimpleImputer(strategy=strategy)
                    
                    df_processed[column] = imputer.fit_transform(
                        df[[column]]
                    ).ravel()
                    self.imputers[column] = imputer
                    
        return df_processed
    
    def _choose_strategy(self, series):
        # Remove missing values for analysis
        clean_series = series.dropna()
        
        # Check distribution
        skewness = clean_series.skew()
        
        if abs(skewness) < 0.5:
            return 'mean'
        elif abs(skewness) < 1:
            return 'median'
        else:
            return 'knn'
```

### Outlier Detection and Treatment
```python
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

class OutlierHandler:
    def __init__(self, method='iqr', contamination=0.1):
        self.method = method
        self.contamination = contamination
        self.bounds = {}
        
    def detect_outliers(self, df, columns=None):
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
            
        outliers = pd.DataFrame(index=df.index)
        
        for col in columns:
            if self.method == 'iqr':
                outliers[col] = self._iqr_outliers(df[col])
            elif self.method == 'zscore':
                outliers[col] = self._zscore_outliers(df[col])
            elif self.method == 'isolation':
                outliers[col] = self._isolation_outliers(df[col])
            elif self.method == 'lof':
                outliers[col] = self._lof_outliers(df[col])
                
        return outliers
    
    def _iqr_outliers(self, series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        self.bounds[series.name] = (lower_bound, upper_bound)
        
        return (series < lower_bound) | (series > upper_bound)
    
    def _zscore_outliers(self, series, threshold=3):
        z_scores = np.abs(stats.zscore(series.dropna()))
        return pd.Series(z_scores > threshold, index=series.dropna().index)
    
    def _isolation_outliers(self, series):
        clf = IsolationForest(contamination=self.contamination, random_state=42)
        outliers = clf.fit_predict(series.values.reshape(-1, 1))
        return pd.Series(outliers == -1, index=series.index)
    
    def treat_outliers(self, df, outliers, method='clip'):
        df_treated = df.copy()
        
        for col in outliers.columns:
            if method == 'clip' and col in self.bounds:
                lower, upper = self.bounds[col]
                df_treated[col] = df_treated[col].clip(lower, upper)
            elif method == 'remove':
                df_treated.loc[outliers[col], col] = np.nan
            elif method == 'transform':
                # Log transformation for positive values
                if (df_treated[col] > 0).all():
                    df_treated[col] = np.log1p(df_treated[col])
                    
        return df_treated
```

## 🔄 Feature Transformation

### Scaling and Normalization
```python
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, 
                                   RobustScaler, QuantileTransformer,
                                   PowerTransformer)

class FeatureScaler:
    def __init__(self, method='auto', feature_range=(0, 1)):
        self.method = method
        self.feature_range = feature_range
        self.scalers = {}
        
    def fit_transform(self, df):
        df_scaled = df.copy()
        
        for column in df.select_dtypes(include=[np.number]).columns:
            if self.method == 'auto':
                scaler = self._choose_scaler(df[column])
            else:
                scaler = self._get_scaler(self.method)
                
            df_scaled[column] = scaler.fit_transform(
                df[[column]]
            ).ravel()
            self.scalers[column] = scaler
            
        return df_scaled
    
    def _choose_scaler(self, series):
        # Check for outliers
        z_scores = np.abs(stats.zscore(series.dropna()))
        has_outliers = (z_scores > 3).any()
        
        # Check distribution
        skewness = series.skew()
        
        if has_outliers:
            return RobustScaler()
        elif abs(skewness) > 1:
            return QuantileTransformer(output_distribution='uniform')
        else:
            return StandardScaler()
    
    def _get_scaler(self, method):
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(feature_range=self.feature_range),
            'robust': RobustScaler(),
            'quantile': QuantileTransformer(),
            'power': PowerTransformer(method='yeo-johnson')
        }
        return scalers.get(method, StandardScaler())
```

### Encoding Categorical Variables
```python
from sklearn.preprocessing import (LabelEncoder, OneHotEncoder, 
                                   OrdinalEncoder, TargetEncoder)
from category_encoders import BinaryEncoder, HashingEncoder

class CategoricalEncoder:
    def __init__(self, strategy='auto', max_cardinality=50):
        self.strategy = strategy
        self.max_cardinality = max_cardinality
        self.encoders = {}
        
    def fit_transform(self, X, y=None):
        X_encoded = X.copy()
        
        for column in X.select_dtypes(include=['object', 'category']).columns:
            cardinality = X[column].nunique()
            
            if self.strategy == 'auto':
                encoder = self._choose_encoder(X[column], cardinality, y)
            else:
                encoder = self._get_encoder(self.strategy)
            
            if isinstance(encoder, OneHotEncoder):
                # Handle one-hot encoding separately
                encoded = encoder.fit_transform(X[[column]])
                feature_names = encoder.get_feature_names_out([column])
                
                # Create DataFrame with encoded features
                encoded_df = pd.DataFrame(
                    encoded.toarray(),
                    columns=feature_names,
                    index=X.index
                )
                
                # Drop original column and concat encoded
                X_encoded = X_encoded.drop(column, axis=1)
                X_encoded = pd.concat([X_encoded, encoded_df], axis=1)
                
            else:
                X_encoded[column] = encoder.fit_transform(X[[column]], y)
                
            self.encoders[column] = encoder
            
        return X_encoded
    
    def _choose_encoder(self, series, cardinality, y):
        if cardinality == 2:
            return LabelEncoder()
        elif cardinality <= 10:
            return OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        elif cardinality <= self.max_cardinality:
            if y is not None:
                return TargetEncoder()
            else:
                return BinaryEncoder()
        else:
            return HashingEncoder(n_components=32)
    
    def _get_encoder(self, strategy):
        encoders = {
            'label': LabelEncoder(),
            'onehot': OneHotEncoder(sparse_output=False),
            'ordinal': OrdinalEncoder(),
            'binary': BinaryEncoder(),
            'target': TargetEncoder(),
            'hashing': HashingEncoder()
        }
        return encoders.get(strategy, LabelEncoder())
```

## 🛠️ Feature Engineering

### Automated Feature Creation
```python
import featuretools as ft
from sklearn.preprocessing import PolynomialFeatures

class FeatureEngineer:
    def __init__(self):
        self.created_features = []
        
    def create_polynomial_features(self, df, degree=2, 
                                  include_columns=None):
        if include_columns is None:
            include_columns = df.select_dtypes(include=[np.number]).columns
            
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_features = poly.fit_transform(df[include_columns])
        
        feature_names = poly.get_feature_names_out(include_columns)
        poly_df = pd.DataFrame(poly_features, columns=feature_names, 
                              index=df.index)
        
        # Remove original features (already in df)
        new_features = [col for col in poly_df.columns 
                       if col not in include_columns]
        
        self.created_features.extend(new_features)
        return pd.concat([df, poly_df[new_features]], axis=1)
    
    def create_interaction_features(self, df, columns=None):
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
            
        interaction_df = pd.DataFrame(index=df.index)
        
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                # Multiplication
                feature_name = f"{col1}_x_{col2}"
                interaction_df[feature_name] = df[col1] * df[col2]
                
                # Division (avoid division by zero)
                if (df[col2] != 0).all():
                    feature_name = f"{col1}_div_{col2}"
                    interaction_df[feature_name] = df[col1] / df[col2]
                    
        self.created_features.extend(interaction_df.columns.tolist())
        return pd.concat([df, interaction_df], axis=1)
    
    def create_aggregate_features(self, df, group_columns, 
                                 agg_columns, agg_funcs=['mean', 'std']):
        agg_df = pd.DataFrame(index=df.index)
        
        for group_col in group_columns:
            for agg_col in agg_columns:
                for func in agg_funcs:
                    feature_name = f"{agg_col}_{func}_by_{group_col}"
                    
                    # Calculate aggregation
                    agg_values = df.groupby(group_col)[agg_col].transform(func)
                    agg_df[feature_name] = agg_values
                    
                    # Ratio to group aggregate
                    ratio_name = f"{agg_col}_ratio_to_{func}_{group_col}"
                    agg_df[ratio_name] = df[agg_col] / (agg_values + 1e-8)
                    
        self.created_features.extend(agg_df.columns.tolist())
        return pd.concat([df, agg_df], axis=1)
    
    def create_time_features(self, df, datetime_columns):
        time_df = pd.DataFrame(index=df.index)
        
        for col in datetime_columns:
            # Convert to datetime if needed
            dt_series = pd.to_datetime(df[col])
            
            # Extract components
            time_df[f"{col}_year"] = dt_series.dt.year
            time_df[f"{col}_month"] = dt_series.dt.month
            time_df[f"{col}_day"] = dt_series.dt.day
            time_df[f"{col}_dayofweek"] = dt_series.dt.dayofweek
            time_df[f"{col}_hour"] = dt_series.dt.hour
            time_df[f"{col}_minute"] = dt_series.dt.minute
            
            # Cyclical encoding
            time_df[f"{col}_month_sin"] = np.sin(2 * np.pi * dt_series.dt.month / 12)
            time_df[f"{col}_month_cos"] = np.cos(2 * np.pi * dt_series.dt.month / 12)
            
            time_df[f"{col}_day_sin"] = np.sin(2 * np.pi * dt_series.dt.day / 31)
            time_df[f"{col}_day_cos"] = np.cos(2 * np.pi * dt_series.dt.day / 31)
            
            # Is weekend
            time_df[f"{col}_is_weekend"] = (dt_series.dt.dayofweek >= 5).astype(int)
            
        self.created_features.extend(time_df.columns.tolist())
        return pd.concat([df, time_df], axis=1)
```

## 📊 Text Preprocessing

### NLP Pipeline
```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import spacy

class TextPreprocessor:
    def __init__(self, language='english'):
        self.language = language
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words(language))
        
        # Load spaCy model
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            import subprocess
            subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
            self.nlp = spacy.load('en_core_web_sm')
    
    def clean_text(self, text):
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_and_normalize(self, text, method='lemma'):
        # Clean text first
        text = self.clean_text(text)
        
        # Tokenize
        tokens = text.split()
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        
        # Normalize
        if method == 'lemma':
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        elif method == 'stem':
            tokens = [self.stemmer.stem(token) for token in tokens]
            
        return tokens
    
    def extract_features(self, texts, method='tfidf', max_features=1000):
        if method == 'tfidf':
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                preprocessor=self.clean_text,
                tokenizer=self.tokenize_and_normalize,
                ngram_range=(1, 2)
            )
        elif method == 'count':
            vectorizer = CountVectorizer(
                max_features=max_features,
                preprocessor=self.clean_text,
                tokenizer=self.tokenize_and_normalize
            )
            
        features = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        return features, feature_names, vectorizer
    
    def extract_entities(self, text):
        doc = self.nlp(text)
        
        entities = {
            'persons': [],
            'organizations': [],
            'locations': [],
            'dates': [],
            'money': []
        }
        
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                entities['persons'].append(ent.text)
            elif ent.label_ == 'ORG':
                entities['organizations'].append(ent.text)
            elif ent.label_ in ['GPE', 'LOC']:
                entities['locations'].append(ent.text)
            elif ent.label_ == 'DATE':
                entities['dates'].append(ent.text)
            elif ent.label_ == 'MONEY':
                entities['money'].append(ent.text)
                
        return entities
```

## 🖼️ Image Preprocessing

### Computer Vision Pipeline
```python
import cv2
from PIL import Image
import torchvision.transforms as transforms
from albumentations import (
    Compose, RandomCrop, HorizontalFlip, VerticalFlip,
    RandomBrightnessContrast, ShiftScaleRotate, 
    Normalize, Resize, CoarseDropout
)

class ImagePreprocessor:
    def __init__(self, target_size=(224, 224), augment=True):
        self.target_size = target_size
        self.augment = augment
        
        # Define augmentation pipeline
        self.train_transform = Compose([
            Resize(target_size[0], target_size[1]),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.1),
            RandomBrightnessContrast(p=0.2),
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, 
                           rotate_limit=15, p=0.2),
            CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2),
            Normalize(mean=[0.485, 0.456, 0.406], 
                     std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = Compose([
            Resize(target_size[0], target_size[1]),
            Normalize(mean=[0.485, 0.456, 0.406], 
                     std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_path, training=True):
        # Read image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        if training and self.augment:
            transformed = self.train_transform(image=image)
        else:
            transformed = self.val_transform(image=image)
            
        return transformed['image']
    
    def extract_color_features(self, image):
        features = {}
        
        # Color histograms
        for i, color in enumerate(['blue', 'green', 'red']):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            features[f'{color}_mean'] = np.mean(image[:, :, i])
            features[f'{color}_std'] = np.std(image[:, :, i])
            
        # HSV features
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        features['hue_mean'] = np.mean(hsv[:, :, 0])
        features['saturation_mean'] = np.mean(hsv[:, :, 1])
        features['value_mean'] = np.mean(hsv[:, :, 2])
        
        return features
    
    def extract_texture_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        features = {}
        
        # Gabor filters
        kernels = []
        for theta in range(4):
            theta = theta / 4. * np.pi
            for sigma in (1, 3):
                for frequency in (0.05, 0.25):
                    kernel = cv2.getGaborKernel((5, 5), sigma, theta, 
                                              frequency, 0.5, 0, 
                                              ktype=cv2.CV_32F)
                    kernels.append(kernel)
        
        # Apply Gabor filters
        for i, kernel in enumerate(kernels):
            filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
            features[f'gabor_{i}_mean'] = np.mean(filtered)
            features[f'gabor_{i}_var'] = np.var(filtered)
            
        return features
```

## 🚀 Pipeline Integration

### Complete Preprocessing Pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class PreprocessingPipeline:
    def __init__(self):
        self.numeric_features = []
        self.categorical_features = []
        self.text_features = []
        self.pipeline = None
        
    def create_pipeline(self, df):
        # Identify feature types
        self.numeric_features = df.select_dtypes(
            include=['int64', 'float64']).columns.tolist()
        self.categorical_features = df.select_dtypes(
            include=['object', 'category']).columns.tolist()
        
        # Create transformers
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False))
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', 
                                     fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        self.pipeline = preprocessor
        return self.pipeline
    
    def fit_transform(self, X, y=None):
        return self.pipeline.fit_transform(X, y)
    
    def transform(self, X):
        return self.pipeline.transform(X)
```

## 💡 Best Practices

### Data Validation
```python
import great_expectations as ge

def validate_data(df):
    # Convert to Great Expectations dataset
    ge_df = ge.from_pandas(df)
    
    # Define expectations
    expectations = {
        'age': {
            'min': 0,
            'max': 120,
            'mostly': 0.95
        },
        'income': {
            'min': 0,
            'not_null': True
        },
        'email': {
            'regex': r'^[\w\.-]+@[\w\.-]+\.\w+$'
        }
    }
    
    results = []
    for column, rules in expectations.items():
        if 'min' in rules and 'max' in rules:
            result = ge_df.expect_column_values_to_be_between(
                column, rules['min'], rules['max'],
                mostly=rules.get('mostly', 1.0)
            )
            results.append(result)
            
        if 'not_null' in rules:
            result = ge_df.expect_column_values_to_not_be_null(column)
            results.append(result)
            
        if 'regex' in rules:
            result = ge_df.expect_column_values_to_match_regex(
                column, rules['regex']
            )
            results.append(result)
    
    return results
```

---

*Master data preprocessing to unlock the full potential of your AI models* 🔄🎯