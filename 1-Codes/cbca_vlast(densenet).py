
# --- 1. SETUP AND INSTALLATION ---
import os
# [MODIFICATION] Set legacy Keras backend for compatibility with keras-cv-attention-models.
# This MUST be set BEFORE importing TensorFlow, otherwise TF 2.16+ loads Keras 3
# and the keras-cv-attention-models package will fail to import.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# === [REPRODUCIBILITY] Environment variables (must be set before importing TensorFlow) ===
os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
# === END OF ENVIRONMENT VARIABLES ===

from google.colab import drive
drive.mount('/content/drive')

# Install the library required for EfficientViT in TensorFlow/Keras, plus xlsxwriter
!pip install -U keras-cv-attention-models xlsxwriter

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import applications, layers, models, optimizers
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, f1_score, log_loss
import matplotlib.pyplot as plt
import itertools
from pathlib import Path
import warnings
import gc
import time
import psutil
import platform
import seaborn as sns
import random
import math
import hashlib
import json
from datetime import datetime

# [MODIFICATION] Model-specific import for EfficientViT-B0 (Transformer-based hybrid backbone)
from keras_cv_attention_models import efficientvit

# Enable TensorFlow op determinism (TF >= 2.8)
try:
    tf.config.experimental.enable_op_determinism()
    print("TensorFlow op determinism enabled.")
except AttributeError:
    print("tf.config.experimental.enable_op_determinism() not available; skipping.")

# Master seed used to derive all sub-seeds
MASTER_SEED = 42

warnings.filterwarnings('ignore')

# Ensure reproducibility
tf.random.set_seed(MASTER_SEED)
np.random.seed(MASTER_SEED)
random.seed(MASTER_SEED)

# --- BACKBONE CONFIGURATION ---
# Single point of configuration for the feature extractor backbone.
# Change BACKBONE_NAME to switch architectures. No other code edits are required.
#
# Supported backbones:
#   - Any model exposed by tf.keras.applications (see BACKBONE_REGISTRY below).
#   - 'EfficientViTB0' (Transformer-based hybrid) via the `keras-cv-attention-models` package.
BACKBONE_NAME = 'DenseNet121'

# Registry of backbones directly available in tf.keras.applications.
# Each entry maps a backbone name to its constructor.
BACKBONE_REGISTRY = {
    # MobileNet family (lightweight CNNs)
    'MobileNetV2':        applications.MobileNetV2,
    'MobileNetV3Small':   applications.MobileNetV3Small,
    'MobileNetV3Large':   applications.MobileNetV3Large,
    # DenseNet family
    'DenseNet121':        applications.DenseNet121,
    'DenseNet169':        applications.DenseNet169,
    'DenseNet201':        applications.DenseNet201,
    # Xception
    'Xception':           applications.Xception,
    # ConvNeXt family
    'ConvNeXtTiny':       applications.ConvNeXtTiny,
    'ConvNeXtSmall':      applications.ConvNeXtSmall,
    'ConvNeXtBase':       applications.ConvNeXtBase,
    'ConvNeXtLarge':      applications.ConvNeXtLarge,
    'ConvNeXtXLarge':     applications.ConvNeXtXLarge,
    # EfficientNet family
    'EfficientNetB0':     applications.EfficientNetB0,
    'EfficientNetB1':     applications.EfficientNetB1,
    'EfficientNetB2':     applications.EfficientNetB2,
    'EfficientNetB3':     applications.EfficientNetB3,
    'EfficientNetB4':     applications.EfficientNetB4,
    'EfficientNetV2B0':   applications.EfficientNetV2B0,
    'EfficientNetV2B1':   applications.EfficientNetV2B1,
    'EfficientNetV2B2':   applications.EfficientNetV2B2,
    'EfficientNetV2S':    applications.EfficientNetV2S,
    'EfficientNetV2M':    applications.EfficientNetV2M,
    'EfficientNetV2L':    applications.EfficientNetV2L,
    # ResNet family
    'ResNet50':           applications.ResNet50,
    'ResNet101':          applications.ResNet101,
    'ResNet152':          applications.ResNet152,
    'ResNet50V2':         applications.ResNet50V2,
    'ResNet101V2':        applications.ResNet101V2,
    'ResNet152V2':        applications.ResNet152V2,
    # Inception family
    'InceptionV3':        applications.InceptionV3,
    'InceptionResNetV2':  applications.InceptionResNetV2,
    # VGG family
    'VGG16':              applications.VGG16,
    'VGG19':              applications.VGG19,
    # NASNet
    'NASNetMobile':       applications.NASNetMobile,
    'NASNetLarge':        applications.NASNetLarge,
}

# Backbones not available in tf.keras.applications are handled separately.
EXTERNAL_BACKBONES = {'EfficientViTB0'}

# --- CONFIGURATION AND PATHS ---
NOTEBOOK_NAME = f'{BACKBONE_NAME}-based model'
#NOTEBOOK_NAME = 'AggLCurve_MNv2_CBCA+AllAug'
BASE_DIR = '/content/drive/MyDrive/ds/8020/'
DATA_ROOT = os.path.join(BASE_DIR, 'image')
RESULTS_DIR = os.path.join(BASE_DIR, 'Results/3.1-CBCA+AllAug', NOTEBOOK_NAME)
MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, 'best_model.h5')
EXCEL_PATH = os.path.join(RESULTS_DIR, f'CBCA-Results_{NOTEBOOK_NAME}.xlsx')

# Hyperparameters
NUM_CLASSES = 3
CLASS_NAMES = ['podo', 'scabies', 'tungiasis']
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
K_SPLITS = 5
AUGMENTATION_FACTOR = 5  # Number of augmented versions per original image
MAJORITY_AUGMENTATION_FACTOR = 2  # Apply 2X augmentation on majority class

# Create necessary directories
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
Path(os.path.join(RESULTS_DIR, 'Plots')).mkdir(parents=True, exist_ok=True)

# --- MEMORY MANAGEMENT UTILITY ---
def cleanup_memory():
    """Clean up memory by closing figures and clearing sessions"""
    plt.close('all')
    tf.keras.backend.clear_session()
    gc.collect()

# --- SYSTEM MONITORING UTILITIES ---
class SystemMonitor:
    """Monitor system resources during training and inference"""

    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.gpu_available = len(tf.config.list_physical_devices('GPU')) > 0

    def start_monitoring(self):
        """Start monitoring session"""
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        return self

    def get_memory_usage(self):
        """Get current memory usage in MB"""
        return psutil.Process().memory_info().rss / 1024 / 1024 - self.start_memory

    def get_elapsed_time(self):
        """Get elapsed time in seconds"""
        return time.time() - self.start_time

    def get_system_info(self):
        """Get system information"""
        system_info = {
            'platform': platform.system(),
            'processor': platform.processor(),
            'gpu_available': self.gpu_available,
            'gpu_name': self._get_gpu_name(),
            'python_version': platform.python_version(),
            'tensorflow_version': tf.__version__
        }
        return system_info

    def _get_gpu_name(self):
        """Get GPU name if available"""
        try:
            if self.gpu_available:
                gpu_devices = tf.config.list_physical_devices('GPU')
                if gpu_devices:
                    return gpu_devices[0].name
            return "CPU"
        except:
            return "Unknown"

# --- COMPREHENSIVE MODEL EVALUATION METRICS ---
class ComprehensiveModelEvaluator:
    """Comprehensive model evaluation with extended metrics"""

    def __init__(self, class_names):
        self.class_names = class_names
        self.num_classes = len(class_names)

    def calculate_sensitivity_specificity(self, y_true, y_pred):
        """Calculate sensitivity and specificity for each class"""
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))

        sensitivity = []
        specificity = []

        for i in range(self.num_classes):
            # True Positive: correctly predicted as class i
            tp = cm[i, i]

            # False Negative: actually class i but predicted as other classes
            fn = np.sum(cm[i, :]) - tp

            # True Negative: correctly predicted as not class i
            tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - tp)

            # False Positive: actually not class i but predicted as class i
            fp = np.sum(cm[:, i]) - tp

            # Sensitivity (Recall)
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            sensitivity.append(sens)

            # Specificity
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificity.append(spec)

        return sensitivity, specificity

    def calculate_precision_recall_f1(self, y_true, y_pred):
        """Calculate precision, recall, and F1-score"""
        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True, zero_division=0)

        precision = [report[cls]['precision'] for cls in self.class_names]
        recall = [report[cls]['recall'] for cls in self.class_names]
        f1 = [report[cls]['f1-score'] for cls in self.class_names]

        return precision, recall, f1

    def calculate_auc_scores(self, y_true, y_pred_probs):
        """Calculate AUC scores for ROC and PR curves"""
        y_true_oh = keras.utils.to_categorical(y_true, num_classes=self.num_classes)

        roc_aucs = []
        pr_aucs = []

        for i in range(self.num_classes):
            # ROC AUC
            fpr, tpr, _ = roc_curve(y_true_oh[:, i], y_pred_probs[:, i])
            roc_auc = auc(fpr, tpr)
            roc_aucs.append(roc_auc)

            # PR AUC
            precision, recall, _ = precision_recall_curve(y_true_oh[:, i], y_pred_probs[:, i])
            pr_auc = auc(recall, precision)
            pr_aucs.append(pr_auc)

        return roc_aucs, pr_aucs

    def measure_inference_speed(self, model, X_data, num_samples=100):
        """Measure inference speed in samples per second"""
        # Ensure num_samples is not more than available data
        num_samples_actual = min(num_samples, len(X_data))
        if num_samples_actual == 0: # Handle empty data case
            return 0, 0

        # Warm-up (predict on a small batch)
        warmup_samples = min(BATCH_SIZE, num_samples_actual)
        _ = model.predict(X_data[:warmup_samples], verbose=0)

        # Actual measurement
        start_time = time.time()
        _ = model.predict(X_data[:num_samples_actual], verbose=0)
        end_time = time.time()

        inference_time = end_time - start_time
        samples_per_second = num_samples_actual / inference_time if inference_time > 0 else 0

        return samples_per_second, inference_time

    def get_model_size_info(self, model):
        """Get model size information"""
        # Save model temporarily to get file size
        temp_path = os.path.join(RESULTS_DIR, 'temp_model.h5')
        model.save(temp_path)

        # Get file size
        file_size_mb = os.path.getsize(temp_path) / (1024 * 1024)

        # Count parameters
        trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
        total_params = trainable_params + non_trainable_params

        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return {
            'file_size_mb': file_size_mb,
            'trainable_params': trainable_params,
            'non_trainable_params': non_trainable_params,
            'total_params': total_params
        }

    def comprehensive_evaluation(self, model, X_data, y_true, y_pred, y_pred_probs, training_time=None):
        """Comprehensive model evaluation"""

        # Basic metrics
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        categorical_loss = log_loss(y_true, y_pred_probs)

        # Extended metrics
        sensitivity, specificity = self.calculate_sensitivity_specificity(y_true, y_pred)
        precision, recall, f1_scores = self.calculate_precision_recall_f1(y_true, y_pred)
        roc_aucs, pr_aucs = self.calculate_auc_scores(y_true, y_pred_probs)

        # Performance metrics
        inference_speed, inference_time = self.measure_inference_speed(model, X_data) # Pass X_data directly
        model_size_info = self.get_model_size_info(model)

        # Memory usage
        memory_usage_mb = psutil.Process().memory_info().rss / 1024 / 1024

        # Compile results
        results = {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'log_loss': categorical_loss,
            'inference_speed_sps': inference_speed,
            'inference_time_seconds': inference_time,
            'training_time_seconds': training_time,
            'memory_usage_mb': memory_usage_mb,
            'model_file_size_mb': model_size_info['file_size_mb'],
            'total_parameters': model_size_info['total_params'],
            'trainable_parameters': model_size_info['trainable_params'],
            'non_trainable_parameters': model_size_info['non_trainable_params']
        }

        # Add per-class metrics
        for i, class_name in enumerate(self.class_names):
            results[f'{class_name}_sensitivity'] = sensitivity[i]
            results[f'{class_name}_specificity'] = specificity[i]
            results[f'{class_name}_precision'] = precision[i]
            results[f'{class_name}_recall'] = recall[i]
            results[f'{class_name}_f1'] = f1_scores[i]
            results[f'{class_name}_roc_auc'] = roc_aucs[i]
            results[f'{class_name}_pr_auc'] = pr_aucs[i]

        # Add average AUC scores
        results['mean_roc_auc'] = np.mean(roc_aucs)
        results['mean_pr_auc'] = np.mean(pr_aucs)
        results['mean_sensitivity'] = np.mean(sensitivity)
        results['mean_specificity'] = np.mean(specificity)

        return results

# --- 4. DATA LOADING UTILITIES ---
def load_data_paths(data_dir):
    """Loads all image paths and labels from the directory structure."""
    filepaths = []
    labels = []

    for class_name in CLASS_NAMES:
        class_path = os.path.join(data_dir, class_name)
        if not os.path.exists(class_path):
            print(f"Warning: Class path not found: {class_path}")
            continue

        # [REPRODUCIBILITY] Sort filenames to guarantee a stable order across runs
        class_files = sorted([
            f for f in os.listdir(class_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        for filename in class_files:
            filepaths.append(os.path.join(class_path, filename))
            labels.append(class_name)

    df = pd.DataFrame({'filepath': filepaths, 'label': labels})
    df['label_idx'] = df['label'].map({name: i for i, name in enumerate(CLASS_NAMES)})
    print(f"Loaded {len(df)} samples from {data_dir}.")
    return df

df_train = load_data_paths(os.path.join(DATA_ROOT, 'train'))
df_test = load_data_paths(os.path.join(DATA_ROOT, 'test'))

# Initialize comprehensive evaluator
model_evaluator = ComprehensiveModelEvaluator(CLASS_NAMES)
system_monitor = SystemMonitor()

# --- CLASS-BASED CONDITIONAL AUGMENTATION FUNCTION ---
def get_class_based_augmentation_datagen(seed=None):
    """Returns Keras ImageDataGenerator with standard geometric augmentation for class-based conditional augmentation.

    [REPRODUCIBILITY] The `seed` argument controls the RNG used for the geometric transforms.
    Passing the same seed produces identical transforms across runs.
    """
    return keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

def balance_dataframe_with_conditional_augmentation(df_fold_train):
    """Balance the training dataframe using class-based conditional augmentation (oversampling)."""

    # Calculate per-fold class distribution
    fold_class_counts = df_fold_train['label'].value_counts()
    per_fold_majority_count = fold_class_counts['scabies']  # Dynamic per-fold majority count

    print(f"Per-fold class distribution: {fold_class_counts.to_dict()}")
    print(f"Per-fold majority count (scabies): {per_fold_majority_count}")

    # Separate classes
    df_minority_podo = df_fold_train[df_fold_train['label'] == 'podo']
    df_minority_tungiasis = df_fold_train[df_fold_train['label'] == 'tungiasis']
    df_majority_scabies = df_fold_train[df_fold_train['label'] == 'scabies']

    # --- STEP 1: First augment the majority class by 2X ---
    target_scabies = per_fold_majority_count * (1 + MAJORITY_AUGMENTATION_FACTOR)  # 3X total (original + 2X augmented)
    N_scabies = len(df_majority_scabies)
    if N_scabies > 0:
        factor_scabies = max(1, int(np.ceil(target_scabies / N_scabies)))
        df_scabies_oversampled = pd.concat([df_majority_scabies] * factor_scabies, ignore_index=True)
        # Trim to target_scabies count
        df_scabies_oversampled = df_scabies_oversampled.sample(n=int(target_scabies), replace=True).reset_index(drop=True)
        print(f"Scabies (majority): {N_scabies} → {target_scabies} (oversampled {factor_scabies}x, 2X augmentation applied)")
    else:
        df_scabies_oversampled = pd.DataFrame(columns=df_fold_train.columns)
        print("Warning: No scabies samples in this fold")

    # --- STEP 2: Now augment minority classes based on the AUGMENTED majority count ---
    augmented_majority_count = target_scabies  # This is the new baseline for ratios

    # Podo replication - 50% of AUGMENTED majority count
    target_podo = augmented_majority_count // 2  # Half of AUGMENTED majority count
    N_podo = len(df_minority_podo)
    if N_podo > 0:
        factor_podo = max(1, int(np.ceil(target_podo / N_podo)))
        df_podo_oversampled = pd.concat([df_minority_podo] * factor_podo, ignore_index=True)
        # Trim to target_podo count
        df_podo_oversampled = df_podo_oversampled.sample(n=int(target_podo), replace=True).reset_index(drop=True)
        print(f"Podo: {N_podo} → {target_podo} (oversampled {factor_podo}x - 50% of augmented majority)")
    else:
        df_podo_oversampled = pd.DataFrame(columns=df_fold_train.columns)
        print("Warning: No podo samples in this fold")

    # Tungiasis replication - 100% of AUGMENTED majority count
    target_tungiasis = augmented_majority_count  # 1:1 with AUGMENTED majority
    N_tungiasis = len(df_minority_tungiasis)
    if N_tungiasis > 0:
        factor_tungiasis = max(1, int(np.ceil(target_tungiasis / N_tungiasis)))
        df_tungiasis_oversampled = pd.concat([df_minority_tungiasis] * factor_tungiasis, ignore_index=True)
        # Trim to target_tungiasis count
        df_tungiasis_oversampled = df_tungiasis_oversampled.sample(n=int(target_tungiasis), replace=True).reset_index(drop=True)
        print(f"Tungiasis: {N_tungiasis} → {target_tungiasis} (oversampled {factor_tungiasis}x - 100% of augmented majority)")
    else:
        df_tungiasis_oversampled = pd.DataFrame(columns=df_fold_train.columns)
        print("Warning: No tungiasis samples in this fold")

    # Combine all with per-fold dynamic counts
    df_balanced_train = pd.concat([
        df_scabies_oversampled,                    # Majority oversampled to 3X (original + 2X augmented)
        df_podo_oversampled,                       # Podo oversampled to 50% of AUGMENTED majority
        df_tungiasis_oversampled                   # Tungiasis oversampled to 100% of AUGMENTED majority
    ], ignore_index=True).sample(frac=1).reset_index(drop=True)  # Final shuffle

    final_counts = df_balanced_train['label'].value_counts()
    print(f"Balanced training set - Final counts: {final_counts.to_dict()}")
    print(f"Total samples: {len(df_balanced_train)}\n")

    return df_balanced_train

# ======================================================================
# *** [MODIFICATION] GENERIC BACKBONE FACTORY ***
# ======================================================================
def _create_efficientvit_b0_feature_extractor(image_size=IMAGE_SIZE):
    """Loads EfficientViTB0 (Transformer-based hybrid backbone) as a Keras feature extractor.

    [INTEGRATED FROM 3_2_3_effvitb0_cbca+allaug+cweights_tial3.py]
    Uses the `keras-cv-attention-models` package (by leondgarse), which provides a
    mature, well-maintained Keras/TensorFlow implementation of EfficientViT-B0.
    This is the correct package for EfficientViT-B0 in Keras — the MIT HAN Lab
    `efficientvit` package is PyTorch-first and its TF support is unofficial.

    IMPORTANT: This function requires `TF_USE_LEGACY_KERAS=1` to be set BEFORE
    importing TensorFlow, because `keras-cv-attention-models` is built against
    the Keras 2.x API (tf.keras) and is NOT compatible with Keras 3. This is
    already configured at the top of the script.

    Installation:
        !pip install -U keras-cv-attention-models

    Returns:
        A frozen Keras Model that outputs a pooled feature vector for each input
        image (4-D output → GlobalAveragePooling2D; 3-D output → mean over tokens).
    """
    try:
        from keras_cv_attention_models import efficientvit
    except ImportError as e:
        raise ImportError(
            "\n\n"
            "EfficientViTB0 requires the `keras-cv-attention-models` package.\n"
            "Install it with:\n"
            "    !pip install -U keras-cv-attention-models\n"
            "Note: this package requires `TF_USE_LEGACY_KERAS=1` to be set BEFORE\n"
            "importing TensorFlow. This is already configured at the top of the script.\n"
        ) from e

    # Build the frozen backbone with the classification head removed.
    # `num_classes=0` is the keras-cv-attention-models equivalent of
    # `include_top=False` used by tf.keras.applications.
    base_model = efficientvit.EfficientViT_B0(
        pretrained='imagenet',
        num_classes=0,                    # Removes the classification head
        input_shape=image_size + (3,),    # Keras input shape: (H, W, C)
    )

    # Freeze the backbone (feature extractor only, no fine-tuning).
    base_model.trainable = False

    # Handle the case where the model exposes a list/tuple of outputs.
    # (Some EfficientViT variants return multiple tensors when num_classes=0.)
    if isinstance(base_model.output, (list, tuple)):
        x = base_model.output[-1]
    else:
        x = base_model.output

    # Apply pooling depending on the output rank:
    #   4-D (B, H, W, C) → spatial feature map → GlobalAveragePooling2D
    #   3-D (B, N, C)    → sequence of tokens  → mean over the token dimension
    if len(x.shape) == 4:
        x = GlobalAveragePooling2D()(x)
    elif len(x.shape) == 3:
        x = layers.Lambda(lambda t: tf.reduce_mean(t, axis=1))(x)
    else:
        raise ValueError(f"Unexpected EfficientViT-B0 output shape: {x.shape}")

    feature_extractor = Model(inputs=base_model.input, outputs=x)
    print("EfficientViTB0 loaded via keras_cv_attention_models.efficientvit.EfficientViT_B0")
    return feature_extractor


def create_feature_extractor(backbone_name, image_size=IMAGE_SIZE):
    """Generic feature extractor factory.

    Supports:
    - Any model in `BACKBONE_REGISTRY` (i.e., tf.keras.applications).
    - 'EfficientViTB0' (Transformer-based hybrid) via the `keras-cv-attention-models` package.

    All returned extractors are frozen and end with a pooled feature vector.

    Args:
        backbone_name: str, one of the keys in BACKBONE_REGISTRY or 'EfficientViTB0'.
        image_size: (H, W) tuple, default IMAGE_SIZE.

    Returns:
        A frozen Keras Model.
    """
    print(f"\nCreating feature extractor for backbone: '{backbone_name}'")

    # Case 1: standard tf.keras.applications backbone
    if backbone_name in BACKBONE_REGISTRY:
        constructor = BACKBONE_REGISTRY[backbone_name]
        base_model = constructor(
            weights='imagenet',
            include_top=False,
            input_shape=image_size + (3,)
        )
        base_model.trainable = False
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        feature_extractor = Model(inputs=base_model.input, outputs=x)
        print(f"Feature extractor created from tf.keras.applications.{backbone_name}")
        return feature_extractor

    # Case 2: external backbone (EfficientViTB0 via keras-cv-attention-models)
    if backbone_name in EXTERNAL_BACKBONES:
        if backbone_name == 'EfficientViTB0':
            return _create_efficientvit_b0_feature_extractor(image_size)
        raise ValueError(f"External backbone '{backbone_name}' is not implemented.")

    # Case 3: unknown
    raise ValueError(
        f"Unknown backbone: '{backbone_name}'.\n"
        f"Supported tf.keras.applications backbones: {sorted(BACKBONE_REGISTRY.keys())}\n"
        f"Supported external backbones: {sorted(EXTERNAL_BACKBONES)}"
    )
# ======================================================================
# *** END OF GENERIC BACKBONE FACTORY ***
# ======================================================================

# --- 5. BASELINE MLP HEAD DEFINITION ---
def create_baseline_model(input_shape=(2048,)):
    """Creates the custom baseline model from the second code for classification.

    Note: input_shape is supplied dynamically at call time from the actual
    feature dimensionality of the chosen backbone, so this default is only a
    placeholder.
    """
    model = Sequential([
        # First convolutional block (adapted for 1D feature vector)
        Dense(512, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.25),

        # Second dense block
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.25),

        # Third dense block
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.25),

        # Output layer
        Dense(NUM_CLASSES, activation='softmax')
    ])

    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer,
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    return model

def extract_features_with_conditional_augmentation(feature_extractor, dataframe, seed=None):
    """Extract features using the chosen backbone with class-based conditional augmentation.

    [MODIFICATION] Applies geometric augmentation via `get_class_based_augmentation_datagen()`.
    [REPRODUCIBILITY] `seed` controls the augmentation RNG; same seed = same transforms.
    """
    features = []
    labels = []

    # First, balance the dataframe using class-based conditional augmentation
    df_balanced = balance_dataframe_with_conditional_augmentation(dataframe)

    # [MODIFICATION] Use geometric augmentation datagen (seeded for reproducibility)
    augmentation_datagen = get_class_based_augmentation_datagen(seed=seed)

    print(f"Extracting features from balanced dataset (WITH GEOMETRIC AUGMENTATION, seed={seed})...")

    # [MODIFICATION] shuffle=True so a fresh transform is applied per batch element
    generator = augmentation_datagen.flow_from_dataframe(
        dataframe=df_balanced,
        x_col='filepath',
        y_col='label',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=seed
    )

    # Extract features batch by batch
    for i in range(len(generator)):
        x_batch, y_batch = next(generator)
        features_batch = feature_extractor.predict(x_batch, verbose=0)
        features.extend(features_batch)
        labels.extend(np.argmax(y_batch, axis=1))

        if (i + 1) * BATCH_SIZE >= len(df_balanced):
            break

    return np.array(features), np.array(labels)

def extract_features(feature_extractor, dataframe, use_conditional_augmentation=False, seed=None):
    """Extract features using the chosen backbone for all images in dataframe.

    [REPRODUCIBILITY] `seed` is forwarded to the augmentation datagen when applicable.
    """
    if use_conditional_augmentation:
        return extract_features_with_conditional_augmentation(feature_extractor, dataframe, seed=seed)

    # Original feature extraction without augmentation
    features = []
    labels = []

    datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_dataframe(
        dataframe=dataframe,
        x_col='filepath',
        y_col='label',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    # Extract features batch by batch
    for i in range(len(generator)):
        x_batch, y_batch = next(generator)
        features_batch = feature_extractor.predict(x_batch, verbose=0)
        features.extend(features_batch)
        labels.extend(np.argmax(y_batch, axis=1))

        if (i + 1) * BATCH_SIZE >= len(dataframe):
            break

    return np.array(features), np.array(labels)

# --- 6. VISUALIZATION UTILITIES FROM SECOND CODE ---
def visualize_training_results(history, fold_id):
    """Plots training and validation loss/accuracy curves."""
    plt.style.use('fivethirtyeight')
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 8))

    # Plot loss
    ax1.plot(history.history['loss'], color='r', label='Training Loss')
    ax1.plot(history.history['val_loss'], color='g', label='Validation Loss')
    ax1.set_title(f'Training and Validation Loss - Fold {fold_id}')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot accuracy
    ax2.plot(history.history['accuracy'], color='r', label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], color='g', label='Validation Accuracy')
    ax2.set_title(f'Training and Validation Accuracy - Fold {fold_id}')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'Plots', f'Learning_{NOTEBOOK_NAME}_Fold_{fold_id}.png'))
    plt.close()

# ========== AGGREGATE LEARNING CURVES ==========

def plot_aggregate_learning_curves(histories, notebook_name, results_dir):
    """
    Plots mean and standard deviation of training/validation loss and accuracy
    across all cross-validation folds, handling variable epoch lengths (due to early stopping).
    """
    if not histories:
        print("No histories to aggregate.")
        return

    # Find maximum number of epochs across all folds
    max_epochs = max(len(h.history['loss']) for h in histories)

    # Pad each history's metrics to max_epochs by repeating the last value
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for hist in histories:
        n_epochs = len(hist.history['loss'])
        # Pad with last value
        train_loss = hist.history['loss'] + [hist.history['loss'][-1]] * (max_epochs - n_epochs)
        val_loss = hist.history['val_loss'] + [hist.history['val_loss'][-1]] * (max_epochs - n_epochs)
        train_acc = hist.history['accuracy'] + [hist.history['accuracy'][-1]] * (max_epochs - n_epochs)
        val_acc = hist.history['val_accuracy'] + [hist.history['val_accuracy'][-1]] * (max_epochs - n_epochs)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

    # Convert to numpy arrays (now all have same length)
    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)
    train_accs = np.array(train_accs)
    val_accs = np.array(val_accs)

    # Compute mean and std across folds
    mean_train_loss = np.mean(train_losses, axis=0)
    std_train_loss = np.std(train_losses, axis=0)
    mean_val_loss = np.mean(val_losses, axis=0)
    std_val_loss = np.std(val_losses, axis=0)
    mean_train_acc = np.mean(train_accs, axis=0)
    std_train_acc = np.std(train_accs, axis=0)
    mean_val_acc = np.mean(val_accs, axis=0)
    std_val_acc = np.std(val_accs, axis=0)

    epochs_range = range(1, max_epochs + 1)

    plt.style.use('fivethirtyeight')
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 8))

    # Loss plot with shaded std region
    ax1.plot(epochs_range, mean_train_loss, color='r', label='Training Loss (mean)')
    ax1.fill_between(epochs_range, mean_train_loss - std_train_loss, mean_train_loss + std_train_loss,
                     color='r', alpha=0.2, label='±1 std')
    ax1.plot(epochs_range, mean_val_loss, color='g', label='Validation Loss (mean)')
    ax1.fill_between(epochs_range, mean_val_loss - std_val_loss, mean_val_loss + std_val_loss,
                     color='g', alpha=0.2, label='±1 std')
    ax1.set_title(f'Aggregate Learning Curves: {notebook_name}\n(Loss)')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Accuracy plot with shaded std region
    ax2.plot(epochs_range, mean_train_acc, color='r', label='Training Accuracy (mean)')
    ax2.fill_between(epochs_range, mean_train_acc - std_train_acc, mean_train_acc + std_train_acc,
                     color='r', alpha=0.2, label='±1 std')
    ax2.plot(epochs_range, mean_val_acc, color='g', label='Validation Accuracy (mean)')
    ax2.fill_between(epochs_range, mean_val_acc - std_val_acc, mean_val_acc + std_val_acc,
                     color='g', alpha=0.2, label='±1 std')
    ax2.set_title(f'Aggregate Learning Curves: {notebook_name}\n(Accuracy)')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    save_path = os.path.join(results_dir, 'Plots', f'Aggregate_Learning_Curves_{notebook_name}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Aggregate learning curves saved to {save_path}")

# ========== END OF NEW FUNCTION ==========


def plot_confusion_matrix(cm, classes, fold_id, normalize=False):
    """Plots the confusion matrix with seaborn style."""
    plt.figure(figsize=(8, 6))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - Fold {fold_id}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'Plots', f'CM_{NOTEBOOK_NAME}_Fold_{fold_id}.png'))
    plt.close()

def plot_roc_pr_curves(y_true_labels, y_pred_probs, fold_id):
    """Plots ROC and Precision-Recall curves."""
    y_true_oh = keras.utils.to_categorical(y_true_labels, num_classes=NUM_CLASSES)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)

    roc_aucs = []
    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_true_oh[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        roc_aucs.append(roc_auc)
        plt.plot(fpr, tpr, label=f'ROC {CLASS_NAMES[i]} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Fold {fold_id}')
    plt.legend(loc="lower right")

    plt.subplot(1, 2, 2)
    pr_aucs = []
    for i in range(NUM_CLASSES):
        precision, recall, _ = precision_recall_curve(y_true_oh[:, i], y_pred_probs[:, i])
        pr_auc = auc(recall, precision)
        pr_aucs.append(pr_auc)
        plt.plot(recall, precision, label=f'PR {CLASS_NAMES[i]} (AUC = {pr_auc:.4f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - Fold {fold_id}')
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'Plots', f'Curves_{NOTEBOOK_NAME}_Fold_{fold_id}.png'))
    plt.close()

    return np.mean(roc_aucs), np.mean(pr_aucs)

# --- 7. RESULTS SAVING ---
def save_results_to_excel(all_fold_metrics, test_metrics, excel_path, notebook_name, comprehensive_metrics=None):
    """Saves detailed metrics and confusion matrices to a multi-sheet Excel file."""
    fold_detail_data = []

    for fold_metrics in all_fold_metrics:
        report = fold_metrics['report']
        fold_detail_data.append({
            'Notebook': notebook_name,
            'Fold': fold_metrics['fold_id'],
            'Val_Loss': fold_metrics['loss'],
            'Val_Accuracy': fold_metrics['accuracy'],
            'Macro_F1': fold_metrics['macro_f1'],
            'Avg_AUC_ROC': fold_metrics['avg_auc_roc'],
            'Avg_AUC_PR': fold_metrics['avg_auc_pr'],
            'C0_Podo_R': report['podo']['recall'],
            'C1_Tungiasis_R': report['tungiasis']['recall'],
            'C2_Scabies_R': report['scabies']['recall'],
            'C0_Podo_F1': report['podo']['f1-score'],
            'C1_Tungiasis_F1': report['tungiasis']['f1-score'],
            'C2_Scabies_F1': report['scabies']['f1-score'],
        })

    df_detail = pd.DataFrame(fold_detail_data)

    # Calculate aggregated statistics for the Summary Sheet
    summary_data = {
        'Metric': ['Val_Accuracy', 'Macro_F1', 'Avg_AUC_ROC', 'Avg_AUC_PR', 'C0_Podo_R', 'C1_Tungiasis_R', 'C2_Scabies_R', 'C0_Podo_F1', 'C1_Tungiasis_F1', 'C2_Scabies_F1'],
        f'{notebook_name}_Mean': [
            df_detail['Val_Accuracy'].mean(), df_detail['Macro_F1'].mean(), df_detail['Avg_AUC_ROC'].mean(), df_detail['Avg_AUC_PR'].mean(),
            df_detail['C0_Podo_R'].mean(), df_detail['C1_Tungiasis_R'].mean(), df_detail['C2_Scabies_R'].mean(),
            df_detail['C0_Podo_F1'].mean(), df_detail['C1_Tungiasis_F1'].mean(), df_detail['C2_Scabies_F1'].mean()
        ],
        f'{notebook_name}_SD': [
            df_detail['Val_Accuracy'].std(), df_detail['Macro_F1'].std(), df_detail['Avg_AUC_ROC'].std(), df_detail['Avg_AUC_PR'].std(),
            df_detail['C0_Podo_R'].std(), df_detail['C1_Tungiasis_R'].std(), df_detail['C2_Scabies_R'].std(),
            df_detail['C0_Podo_F1'].std(), df_detail['C1_Tungiasis_F1'].std(), df_detail['C2_Scabies_F1'].std()
        ]
    }
    df_summary = pd.DataFrame(summary_data)

    # Test Set Evaluation Data
    df_test_metrics = pd.DataFrame([test_metrics])
    df_test_metrics = df_test_metrics.T.reset_index()
    df_test_metrics.columns = ['Metric', 'Value']

    # Comprehensive Metrics Sheet
    df_comprehensive = pd.DataFrame()
    if comprehensive_metrics:
        comprehensive_data = []
        for stage, metrics in comprehensive_metrics.items():
            row = {'Stage': stage}
            row.update(metrics)
            comprehensive_data.append(row)

        df_comprehensive = pd.DataFrame(comprehensive_data)

    # Use Pandas ExcelWriter for multi-sheet output
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        df_summary.to_excel(writer, sheet_name='Summary_CV_Stats', index=False)
        df_detail.to_excel(writer, sheet_name='Fold_Detail_Metrics', index=False)
        df_test_metrics.to_excel(writer, sheet_name='Test_Set_Evaluation', index=False)

        # Add comprehensive metrics sheet
        if not df_comprehensive.empty:
            df_comprehensive.to_excel(writer, sheet_name='Comprehensive_Metrics', index=False)

        # Write Confusion Matrices
        for fold_metrics in all_fold_metrics:
            cm_raw = fold_metrics['confusion_matrix']
            df_cm = pd.DataFrame(cm_raw, index=CLASS_NAMES, columns=CLASS_NAMES)
            df_cm.index.name = 'True Label'
            df_cm.columns.name = 'Predicted Label'
            df_cm.to_excel(writer, sheet_name=f'CM_Fold{fold_metrics["fold_id"]}')

    print(f"\nResults for {notebook_name} saved successfully to {excel_path}.")

# ========== REPRODUCIBILITY FINGERPRINT UTILITY ==========

def save_reproducibility_fingerprint(df_train, df_test, output_dir):
    """Saves all seed values, environment versions, and dataframe hashes for reproducibility verification."""
    fingerprint = {
        'timestamp': datetime.now().isoformat(),
        'master_seed': MASTER_SEED,
        'backbone_name': BACKBONE_NAME,
        'tf_version': tf.__version__,
        'numpy_version': np.__version__,
        'python_version': platform.python_version(),
        'gpu_name': system_monitor._get_gpu_name(),
        'train_df_hash': hashlib.md5(
            pd.util.hash_pandas_object(df_train['filepath']).values.tobytes()
        ).hexdigest(),
        'test_df_hash': hashlib.md5(
            pd.util.hash_pandas_object(df_test['filepath']).values.tobytes()
        ).hexdigest(),
        'train_size': len(df_train),
        'test_size': len(df_test),
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'k_splits': K_SPLITS,
        'majority_aug_factor': MAJORITY_AUGMENTATION_FACTOR,
        'augmentation_params': {
            'rotation_range': 20,
            'width_shift_range': 0.1,
            'height_shift_range': 0.1,
            'shear_range': 0.1,
            'zoom_range': 0.1,
            'horizontal_flip': True,
        }
    }

    fp_path = os.path.join(output_dir, 'reproducibility_fingerprint.json')
    with open(fp_path, 'w') as f:
        json.dump(fingerprint, f, indent=2)
    print(f"\nReproducibility fingerprint saved to {fp_path}")
    print(f"Train dataframe hash: {fingerprint['train_df_hash']}")
    print(f"Test dataframe hash:  {fingerprint['test_df_hash']}")
# ========== END OF REPRODUCIBILITY FINGERPRINT UTILITY ==========

# --- 8. MAIN CV TRAINING LOOP WITH FEATURE EXTRACTION AND CLASS-BASED CONDITIONAL AUGMENTATION ---

def run_cv_training_with_feature_extraction(df_train, df_test, notebook_name, backbone_name=BACKBONE_NAME):
    """
    Executes the 5-fold Stratified Cross-Validation process using the chosen backbone
    with class-based conditional augmentation.
    CORRECTED: Validation sets use ONLY original images, not augmented ones.
    """
    print(f"\n--- Starting {notebook_name}: Feature Extraction with {backbone_name} and Class-Based Conditional Augmentation ---")
    print("CORRECTED: Validation sets use ONLY original images")
    print("MODIFIED: Removed secondary augmentation - only class-based conditional augmentation for balancing")
    print(f"MAJORITY CLASS AUGMENTATION: Applying {MAJORITY_AUGMENTATION_FACTOR}X augmentation on scabies class")

    # [MODIFICATION] Create feature extractor via the generic dispatcher
    feature_extractor = create_feature_extractor(backbone_name, image_size=IMAGE_SIZE)
    print(f"{backbone_name} feature extractor created successfully.")

    # Extract features for test data WITHOUT AUGMENTATION
    print("Extracting features from test data without augmentation...")
    X_test, y_test = extract_features(feature_extractor, df_test, use_conditional_augmentation=False)
    print(f"Test features shape: {X_test.shape}")

    # Initialize variables for CV
    skf = StratifiedKFold(n_splits=K_SPLITS, shuffle=True, random_state=MASTER_SEED)
    all_fold_metrics = []
    best_val_macro_f1 = -1.0
    best_model_overall = None
    comprehensive_metrics = {}
    system_info = system_monitor.get_system_info()

    # === NEW: List to store training histories for aggregate curve ===
    fold_histories = []

    print(f"\n--- Starting {notebook_name}: {K_SPLITS}-Fold SKF Training on {backbone_name} Features with Class-Based Conditional Augmentation ---")
    print(f"System Information: {system_info}")

    # Extract ORIGINAL features for entire training set (for proper cross-validation splits)
    print("Extracting ORIGINAL features from training data for cross-validation...")
    X_train_original, y_train_original = extract_features(feature_extractor, df_train, use_conditional_augmentation=False)
    print(f"Original training features shape: {X_train_original.shape}")

    for fold, (train_index, val_index) in enumerate(skf.split(X_train_original, y_train_original)):
        fold_id = fold + 1
        print(f"\n--- Fold {fold_id}/{K_SPLITS} ---")

        # Get the original data splits
        X_fold_train_original, X_fold_val = X_train_original[train_index], X_train_original[val_index]
        y_fold_train_original, y_fold_val = y_train_original[train_index], y_train_original[val_index]

        # Get the corresponding dataframe rows for training fold to apply class-based conditional augmentation
        df_fold_train = df_train.iloc[train_index].reset_index(drop=True)

        # [REPRODUCIBILITY] Derive a deterministic per-fold seed from MASTER_SEED
        fold_seed = MASTER_SEED + fold_id

        # Extract features with CLASS-BASED CONDITIONAL AUGMENTATION for training fold only
        print(f"Extracting features with CLASS-BASED CONDITIONAL AUGMENTATION for training fold {fold_id} (seed={fold_seed})...")
        X_fold_train_augmented, y_fold_train_augmented = extract_features(
            feature_extractor,
            df_fold_train,
            use_conditional_augmentation=True,
            seed=fold_seed
        )
        print(f"Class-based conditionally augmented training features shape: {X_fold_train_augmented.shape}")
        print(f"Validation features shape: {X_fold_val.shape}")

        # Convert labels to categorical
        y_fold_train_cat = keras.utils.to_categorical(y_fold_train_augmented, num_classes=NUM_CLASSES)
        y_fold_val_cat = keras.utils.to_categorical(y_fold_val, num_classes=NUM_CLASSES)

        # [REPRODUCIBILITY] Re-seed TF and NumPy before each fold so that per-fold
        # model initialization is independent of prior folds' RNG consumption
        fold_model_seed = MASTER_SEED + 1000 + fold_id
        tf.random.set_seed(fold_model_seed)
        np.random.seed(fold_model_seed)
        random.seed(fold_model_seed)
        print(f"Fold {fold_id} model initialization seed: {fold_model_seed}")

        # Create baseline model (MLP head) with dynamic feature dimension
        model = create_baseline_model(input_shape=(X_train_original.shape[1],))

        # Callbacks
        model_checkpoint = ModelCheckpoint(
            filepath=os.path.join(RESULTS_DIR, f'fold_{fold_id}_best_weights.weights.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            mode='min',
            verbose=0
        )
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Train model with timing
        fold_start_time = time.time()
        history = model.fit(
            X_fold_train_augmented, y_fold_train_cat,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_fold_val, y_fold_val_cat),
            callbacks=[model_checkpoint, early_stopping],
            verbose=1
        )
        fold_training_time = time.time() - fold_start_time

        # === Store history for aggregate curve ===
        fold_histories.append(history)

        # Visualize training results
        visualize_training_results(history, fold_id)

        # Evaluate on validation set
        y_pred_probs = model.predict(X_fold_val, verbose=0)
        y_pred_labels = np.argmax(y_pred_probs, axis=1)
        y_true_labels = y_fold_val

        # Calculate comprehensive metrics
        fold_comprehensive_metrics = model_evaluator.comprehensive_evaluation(
            model, X_fold_val, y_true_labels, y_pred_labels, y_pred_probs, fold_training_time
        )

        # Store comprehensive metrics
        comprehensive_metrics[f'Fold_{fold_id}_Validation'] = fold_comprehensive_metrics

        # Calculate traditional metrics
        cm = confusion_matrix(y_true_labels, y_pred_labels)
        report = classification_report(y_true_labels, y_pred_labels, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
        macro_f1 = f1_score(y_true_labels, y_pred_labels, average='macro', zero_division=0)
        avg_auc_roc, avg_auc_pr = plot_roc_pr_curves(y_true_labels, y_pred_probs, fold_id)

        # Evaluate model (loss and accuracy)
        loss, accuracy = model.evaluate(X_fold_val, y_fold_val_cat, verbose=0)

        # Store results
        all_fold_metrics.append({
            'fold_id': fold_id,
            'loss': loss,
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'report': report,
            'macro_f1': macro_f1,
            'avg_auc_roc': avg_auc_roc,
            'avg_auc_pr': avg_auc_pr,
            'model': model,
            'comprehensive_metrics': fold_comprehensive_metrics
        })

        # Plot confusion matrix
        plot_confusion_matrix(cm, CLASS_NAMES, fold_id, normalize=True)

        # Track the best model based on validation Macro F1
        if macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = macro_f1
            best_model_overall = model
            model.save(MODEL_SAVE_PATH)

        # Cleanup memory after each fold
        cleanup_memory()

    # === NEW: Plot aggregate learning curves after all folds ===
    plot_aggregate_learning_curves(fold_histories, notebook_name, RESULTS_DIR)

    # Final Test Set Evaluation
    print(f"\n--- Final Evaluation on Independent Test Set ---")

    if best_model_overall is None:
        print("Loading weights from the last best-performing fold...")
        best_model_overall = create_baseline_model(input_shape=(X_train_original.shape[1],))
        best_model_overall.load_weights(os.path.join(RESULTS_DIR, f'fold_{K_SPLITS}_best_weights.weights.h5'))

    # Convert test labels to categorical
    y_test_cat = keras.utils.to_categorical(y_test, num_classes=NUM_CLASSES)

    # Comprehensive test evaluation
    y_test_probs = best_model_overall.predict(X_test, verbose=0)
    y_test_pred = np.argmax(y_test_probs, axis=1)
    y_test_true = y_test

    # ========== [NEW] SAVE PREDICTIONS FOR LATER STATISTICAL COMPARISON ==========
    # Saves y_true and y_pred for the held-out test set so that a standalone
    # statistics script can run McNemar's test and bootstrap confidence intervals
    # against the equivalent outputs of other experiments (e.g., Hybrid, CWP).
    pred_save_path = os.path.join(RESULTS_DIR, 'test_predictions.npz')
    np.savez(pred_save_path, y_true=y_test_true, y_pred=y_test_pred)
    print(f"Test predictions saved to {pred_save_path}")
    # ========== END OF SAVING SECTION ==========

    # Calculate comprehensive test metrics
    test_comprehensive_metrics = model_evaluator.comprehensive_evaluation(
        best_model_overall, X_test, y_test_true, y_test_pred, y_test_probs
    )
    comprehensive_metrics['Final_Test'] = test_comprehensive_metrics

    # Traditional test metrics
    test_loss, test_accuracy = best_model_overall.evaluate(X_test, y_test_cat, verbose=1)
    test_report = classification_report(y_test_true, y_test_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    test_cm = confusion_matrix(y_test_true, y_test_pred)

    # [EXISTING] Normalized Test Confusion Matrix (KEPT UNCHANGED)
    plot_confusion_matrix(test_cm, CLASS_NAMES, 'Test', normalize=True)

    # ========== [NEW] RAW (UNNORMALIZED) TEST CONFUSION MATRIX ==========
    # This creates a separate plot with raw integer counts, without replacing
    # the existing normalized one. Directly addresses Editorial Comment #18,
    # which requested raw counts given the 13-image podo test subset.
    plot_confusion_matrix(test_cm, CLASS_NAMES, 'Test_Raw', normalize=False)
    print(f"Raw test confusion matrix saved as 'CM_{NOTEBOOK_NAME}_Fold_Test_Raw.png'")
    # ========== END OF RAW CM SECTION ==========

    test_avg_auc_roc, test_avg_auc_pr = plot_roc_pr_curves(y_test_true, y_test_probs, 'Test')

    test_metrics = {
        'Test_Loss': test_loss,
        'Test_Accuracy': test_accuracy,
        'Test_Macro_F1': f1_score(y_test_true, y_test_pred, average='macro', zero_division=0),
        'Test_Avg_AUC_ROC': test_avg_auc_roc,
        'Test_Avg_AUC_PR': test_avg_auc_pr,
    }

    # Save results with comprehensive metrics
    save_results_to_excel(all_fold_metrics, test_metrics, EXCEL_PATH, NOTEBOOK_NAME, comprehensive_metrics)

    print(f"\nNotebook {NOTEBOOK_NAME} finished with comprehensive evaluation.")
    print(f"Best model saved to {MODEL_SAVE_PATH}")
    print(f"Comprehensive metrics saved to Excel report")

    return all_fold_metrics, comprehensive_metrics

# --- 9. EXECUTE THE FEATURE EXTRACTION + BASELINE MODEL TRAINING ---

# Execute the main CV loop with feature extraction
baseline_metrics, comprehensive_metrics = run_cv_training_with_feature_extraction(
    df_train,
    df_test,
    notebook_name=NOTEBOOK_NAME,
    backbone_name=BACKBONE_NAME
)

print("\n=== EXPERIMENT COMPLETED SUCCESSFULLY ===")
print(f"{BACKBONE_NAME} Feature Extraction + Baseline Model with Class-Based Conditional Augmentation finished!")
print(f"Results saved to: {RESULTS_DIR}")
print(f"Excel report saved to: {EXCEL_PATH}")

# ========== [NEW] SAVE REPRODUCIBILITY FINGERPRINT ==========
save_reproducibility_fingerprint(df_train, df_test, RESULTS_DIR)
# ========== END OF REPRODUCIBILITY FINGERPRINT SAVE ==========