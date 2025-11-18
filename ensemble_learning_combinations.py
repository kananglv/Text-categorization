import os
import glob
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Conv1D, GlobalMaxPooling1D, Bidirectional
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from itertools import combinations
from scipy.stats import mode


def load_data_from_separate_folders(train_folder="DATA FOR THE MODEL TRAIN",
                                    test_folder="DATA FOR THE MODEL TEST"):
    """
    Load data from separate training and testing folders.
    Split test folder data into validation and test sets (50/50).
    Returns: X_train, X_val, X_test, y_train, y_val, y_test, label_encoder
    """
    # Load training data
    print(f"Loading training data from {train_folder}...")
    train_texts = []
    train_labels = []

    # Load files from training folder
    for class_dir in os.listdir(train_folder):
        class_path = os.path.join(train_folder, class_dir)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                file_path = os.path.join(class_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                        content = file.read()
                        train_texts.append(content)
                        train_labels.append(class_dir)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    # Load test folder data
    print(f"Loading test data from {test_folder}...")
    test_texts = []
    test_labels = []

    for class_dir in os.listdir(test_folder):
        class_path = os.path.join(test_folder, class_dir)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                file_path = os.path.join(class_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                        content = file.read()
                        test_texts.append(content)
                        test_labels.append(class_dir)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    # Encode labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)

    # Apply same encoding to test labels
    y_test_all = label_encoder.transform(test_labels)

    # Split test data into validation and test sets (50/50)
    X_val, X_test, y_val, y_test = train_test_split(
        test_texts, y_test_all, test_size=0.5, random_state=42, stratify=y_test_all
    )

    print(f"Dataset statistics:")
    print(f"- Training samples: {len(train_texts)}")
    print(f"- Validation samples: {len(X_val)}")
    print(f"- Test samples: {len(X_test)}")
    print(f"- Categories: {len(label_encoder.classes_)}")

    return train_texts, X_val, X_test, y_train, y_val, y_test, label_encoder


def plot_confusion_matrix(y_true, y_pred, class_names=None, title='Confusion Matrix',
                          figsize=None, cmap='Blues', normalize=False, filename=None):
    """
    Create and plot a confusion matrix for a classification model.
    """
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Determine number of classes
    num_classes = cm.shape[0]

    # Use default class names if not provided
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    # Determine figure size based on number of classes
    if figsize is None:
        # Base size of 8x6, but increase for more classes
        if num_classes <= 10:
            figsize = (10, 8)
        else:
            # Scale figure size based on number of classes
            scale_factor = num_classes / 10
            figsize = (10 * scale_factor, 8 * scale_factor)

    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = title + ' (Normalized)'
    else:
        fmt = 'd'

    # Set up figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    im = sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap,
                     xticklabels=class_names, yticklabels=class_names,
                     ax=ax, cbar=True)

    # Adjust font size for annotations based on number of classes
    if num_classes > 20:
        for text in im.texts:
            text.set_fontsize(7)

    # Add labels and title
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(title)

    # Rotate tick labels
    plt.xticks(rotation=45, ha='right')
    if num_classes > 20:
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)

    # Adjust layout
    plt.tight_layout()

    # Save if filename is provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    return fig, cm


def train_svm_wrapper(X_train, X_val, X_test, y_train, y_val, y_test, label_encoder=None):
    """Train SVM model and return predictions with metrics"""
    print("\nTraining SVM model...")
    start = time.time()

    # Create pipeline with best parameters
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=90000, ngram_range=(1, 2))),
        ('clf', CalibratedClassifierCV(
            LinearSVC(C=1.0, dual=False, random_state=42, max_iter=2000),
            method='sigmoid', cv=5))
    ])

    # Train on training data
    pipeline.fit(X_train, y_train)

    # Evaluate on validation data
    val_preds = pipeline.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_preds)
    print(f"  Validation accuracy: {val_accuracy:.4f}")

    # Predict on test set
    y_proba = pipeline.predict_proba(X_test)
    y_pred = pipeline.predict(X_test)

    # Calculate all metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    training_time = time.time() - start

    print(f"SVM Results:")
    print(f"  Test accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Training time: {training_time:.2f} seconds")

    # Plot confusion matrix
    if label_encoder is not None:
        class_names = label_encoder.classes_
    else:
        class_names = [f"Class {i}" for i in range(len(np.unique(y_test)))]

    print("\nGenerating SVM confusion matrix...")
    fig_norm, cm = plot_confusion_matrix(
        y_test, y_pred,
        class_names=class_names,
        title='SVM Confusion Matrix (Normalized)',
        normalize=True,
        filename='svm_confusion_matrix_normalized.png'
    )

    return {
        'name': 'SVM',
        'model': pipeline,
        'predictions': y_pred,
        'probabilities': y_proba,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'training_time': training_time,
        'confusion_matrix': cm
    }


def train_nb_wrapper(X_train, X_val, X_test, y_train, y_val, y_test, label_encoder=None):
    """Train Naive Bayes model and return predictions with metrics"""
    print("\nTraining Naive Bayes model...")
    start = time.time()

    # Create pipeline with best parameters
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=90000, ngram_range=(1, 2), min_df=1)),
        ('clf', MultinomialNB(alpha=0.05, fit_prior=False))
    ])

    # Train on training data
    pipeline.fit(X_train, y_train)

    # Evaluate on validation data
    val_preds = pipeline.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_preds)
    print(f"  Validation accuracy: {val_accuracy:.4f}")

    # Predict on test set
    y_proba = pipeline.predict_proba(X_test)
    y_pred = pipeline.predict(X_test)

    # Calculate all metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    training_time = time.time() - start

    print(f"Naive Bayes Results:")
    print(f"  Test accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Training time: {training_time:.2f} seconds")

    # Plot confusion matrix
    if label_encoder is not None:
        class_names = label_encoder.classes_
    else:
        class_names = [f"Class {i}" for i in range(len(np.unique(y_test)))]

    print("\nGenerating Naive Bayes confusion matrix...")
    fig_norm, cm = plot_confusion_matrix(
        y_test, y_pred,
        class_names=class_names,
        title='Naive Bayes Confusion Matrix (Normalized)',
        normalize=True,
        filename='naive_bayes_confusion_matrix_normalized.png'
    )

    return {
        'name': 'NaiveBayes',
        'model': pipeline,
        'predictions': y_pred,
        'probabilities': y_proba,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'training_time': training_time,
        'confusion_matrix': cm
    }


def train_rf_wrapper(X_train, X_val, X_test, y_train, y_val, y_test, label_encoder=None):
    """Train Random Forest model and return predictions with metrics"""
    print("\nTraining Random Forest model...")
    start = time.time()

    # Create pipeline with best parameters
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=80000)),
        ('clf', RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=5,
            criterion='gini',
            max_features='log2',
            random_state=42,
            n_jobs=-1))
    ])

    # Train on training data
    pipeline.fit(X_train, y_train)

    # Evaluate on validation data
    val_preds = pipeline.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_preds)
    print(f"  Validation accuracy: {val_accuracy:.4f}")

    # Predict on test set
    y_proba = pipeline.predict_proba(X_test)
    y_pred = pipeline.predict(X_test)

    # Calculate all metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    training_time = time.time() - start

    print(f"Random Forest Results:")
    print(f"  Test accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Training time: {training_time:.2f} seconds")

    # Get class names for confusion matrix
    if label_encoder is not None:
        class_names = label_encoder.classes_
    else:
        class_names = [f"Class {i}" for i in range(len(np.unique(y_test)))]

    print("\nGenerating Random Forest confusion matrix...")
    fig_norm, cm = plot_confusion_matrix(
        y_test, y_pred,
        class_names=class_names,
        title='Random Forest Confusion Matrix (Normalized)',
        normalize=True,
        filename='rf_confusion_matrix_normalized.png'
    )

    return {
        'name': 'RandomForest',
        'model': pipeline,
        'predictions': y_pred,
        'probabilities': y_proba,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'training_time': training_time,
        'confusion_matrix': cm
    }


def train_cnn_wrapper(X_train, X_val, X_test, y_train, y_val, y_test, label_encoder=None, max_features=50000,
                      maxlen=500):
    """Train CNN model and return predictions with metrics"""
    print("\nTraining CNN model...")
    start = time.time()

    # Set random seed for reproducibility
    random_state = 42
    tf.random.set_seed(random_state)
    np.random.seed(random_state)

    # Create and fit tokenizer
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(X_train)

    # Convert text to sequences and pad
    X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=maxlen)
    X_val_pad = pad_sequences(tokenizer.texts_to_sequences(X_val), maxlen=maxlen)
    X_test_pad = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=maxlen)

    # Number of classes
    num_classes = len(np.unique(np.concatenate([y_train, y_val, y_test])))

    # Build CNN model with best parameters
    model = Sequential()
    model.add(Embedding(max_features, 1024, input_length=maxlen))
    model.add(Conv1D(128, 5, activation='relu', kernel_regularizer=l2(0.002)))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.002)))
    model.add(Dropout(0.3, seed=random_state))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Define callbacks
    early_stop = EarlyStopping(
        monitor='val_accuracy', mode='max', patience=5, restore_best_weights=True, verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_accuracy', mode='max', factor=0.5, patience=5, min_lr=0.00001, verbose=1
    )

    # Train model with explicit validation data
    model.fit(
        X_train_pad, y_train,
        batch_size=128,
        epochs=30,
        validation_data=(X_val_pad, y_val),
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # Make predictions
    y_proba = model.predict(X_test_pad)
    y_pred = np.argmax(y_proba, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    training_time = time.time() - start

    print(f"CNN Results:")
    print(f"  Test accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Training time: {training_time:.2f} seconds")

    # Get class names for confusion matrix
    if label_encoder is not None:
        class_names = label_encoder.classes_
    else:
        class_names = [f"Class {i}" for i in range(len(np.unique(y_test)))]

    # Generate confusion matrix
    print("\nGenerating CNN confusion matrix...")
    fig_norm, cm = plot_confusion_matrix(
        y_test, y_pred,
        class_names=class_names,
        title='CNN Confusion Matrix (Normalized)',
        normalize=True,
        filename='cnn_confusion_matrix_normalized.png'
    )

    return {
        'name': 'CNN',
        'model': model,
        'tokenizer': tokenizer,
        'predictions': y_pred,
        'probabilities': y_proba,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'training_time': training_time,
        'confusion_matrix': cm
    }

def create_model_comparison_chart(results):
    # Extract model names and metrics
    model_names = [result['model'] for result in results]
    accuracies = [result['accuracy'] for result in results]
    precisions = [result['precision'] for result in results]
    recalls = [result['recall'] for result in results]
    f1_scores = [result['f1'] for result in results]

    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Model': model_names,
        'Accuracy': accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1 Score': f1_scores
    })

    # Sort by accuracy
    df = df.sort_values('Accuracy', ascending=False)

    # Set up the figure
    plt.figure(figsize=(18, 12))

    # Plot accuracy bars
    bars = plt.barh(df['Model'], df['Accuracy'], color='skyblue')

    # Add data labels to the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{width:.4f}', ha='left', va='center')

    # Add gridlines
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Add title and labels
    plt.title('Model Accuracy Comparison', fontsize=12)
    plt.xlabel('Accuracy', fontsize=10)
    plt.ylabel('Model', fontsize=10)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('model_accuracy_comparison.png', dpi=300)
    plt.show()

    print("Model comparison chart created and saved as 'model_accuracy_comparison.png'")

    # Creating a grouped bar chart for all metrics
    plt.figure(figsize=(18, 12))

    # Prepare data for grouped bar chart
    df_melt = pd.melt(df, id_vars=['Model'], value_vars=['Accuracy', 'Precision', 'Recall', 'F1 Score'])

    # Create grouped bar chart
    sns.barplot(x='value', y='Model', hue='variable', data=df_melt)

    # Add title and labels
    plt.title('Model Performance Metrics Comparison', fontsize=14)
    plt.xlabel('Score', fontsize=10)
    plt.ylabel('Model', fontsize=10)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('model_metrics_comparison.png', dpi=300)
    plt.show()

    print("Complete metrics comparison chart created and saved as 'model_metrics_comparison.png'")


def create_soft_voting_ensemble(models_results, y_test, weights=None):
    """Create a soft voting ensemble from the provided models."""
    print("\n" + "=" * 50)
    print("Creating Soft Voting Ensemble...")

    # Extract model names and probabilities
    model_names = [result['name'] for result in models_results]
    probabilities = [result['probabilities'] for result in models_results]

    # If weights are not provided, use model accuracies
    if weights is None:
        weights = [result['accuracy'] for result in models_results]
        print(f"Using model accuracy-based weights: {[f'{w:.4f}' for w in weights]}")
    else:
        print(f"Using custom weights: {[f'{w:.4f}' for w in weights]}")

    # Normalize weights
    weights = np.array(weights) / sum(weights)

    print(f"Using models: {model_names}")
    print(f"With normalized weights: {[f'{w:.4f}' for w in weights]}")

    # Combine probabilities using weights
    weighted_probs = np.zeros_like(probabilities[0])
    for probs, weight in zip(probabilities, weights):
        weighted_probs += weight * probs

    # Get final predictions
    ensemble_preds = np.argmax(weighted_probs, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_test, ensemble_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, ensemble_preds, average='weighted')

    # Print results
    print(f"Soft Voting Ensemble Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")

    return {
        'model': f"Soft Voting ({', '.join(model_names)})",
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'weights': weights,
        'model_names': model_names
    }


def test_all_model_combinations(models_results, y_test):
    """Test all possible combinations of models using different voting strategies."""
    print("\n" + "=" * 50)
    print("Testing All Model Combinations with Multiple Voting Strategies...")

    all_ensemble_results = []

    # Generate all combinations of models (from 2 to all models)
    for r in range(2, len(models_results) + 1):
        for combo in combinations(range(len(models_results)), r):
            # Extract the selected models for this combination
            selected_models = [models_results[i] for i in combo]
            selected_names = [model['name'] for model in selected_models]

            print(f"\nTesting combination: {selected_names}")

            # 1. Soft voting with equal weights
            equal_weights = [1] * len(selected_models)
            soft_equal_result = create_soft_voting_ensemble(selected_models, y_test, weights=equal_weights)
            soft_equal_result['model'] = "Soft-Equal: " + soft_equal_result['model']
            all_ensemble_results.append(soft_equal_result)

            # 2. Soft voting with accuracy weights
            soft_weighted_result = create_soft_voting_ensemble(selected_models, y_test)
            soft_weighted_result['model'] = "Soft-Weighted: " + soft_weighted_result['model']
            all_ensemble_results.append(soft_weighted_result)

    return all_ensemble_results


def optimize_ensemble_weights(models_results, y_test, voting_type='soft', n_trials=100):
    """Optimize model weights for ensemble using random search and generate confusion matrix."""
    print("\n" + "=" * 50)
    print(f"Optimizing {voting_type.capitalize()} Voting Ensemble Weights...")

    best_accuracy = 0
    best_weights = None
    best_result = None
    best_preds = None

    # Make sure models have probability predictions for soft voting
    if voting_type == 'soft' and not all('probabilities' in model for model in models_results):
        print("Not all models provide probability outputs. Cannot perform soft voting optimization.")
        return None

    # Create voting function based on type
    voting_func = create_soft_voting_ensemble if voting_type == 'soft' else create_hard_voting_ensemble

    # Log header
    print(f"{'Trial':<6} {'Weights':<40} {'Accuracy':<10}")
    print("-" * 60)

    # Random search for optimal weights
    for trial in range(n_trials):
        # Generate random weights
        random_weights = np.random.random(len(models_results))

        # Create ensemble with these weights
        result = voting_func(models_results, y_test, weights=random_weights)

        # Track best performance
        if result['accuracy'] > best_accuracy:
            best_accuracy = result['accuracy']
            best_weights = result['weights']
            best_result = result

            # Extract predictions for confusion matrix
            if 'predictions' in result:
                best_preds = result['predictions']
            else:
                # If predictions aren't stored, recalculate them
                probabilities = [model['probabilities'] for model in models_results]
                weighted_probs = np.zeros_like(probabilities[0])
                for probs, weight in zip(probabilities, best_weights):
                    weighted_probs += weight * probs
                best_preds = np.argmax(weighted_probs, axis=1)

        # Log progress (every 10 trials)
        if trial % 10 == 0:
            weights_str = ', '.join([f"{w:.3f}" for w in random_weights])
            print(f"{trial:<6} {weights_str:<40} {result['accuracy']:.6f}")

    # Final best result
    print("\nOptimization Complete!")
    print(f"Best {voting_type.capitalize()} Voting Weights: {[f'{w:.4f}' for w in best_weights]}")
    print(f"Best Accuracy: {best_accuracy:.6f}")

    # Generate confusion matrix for the best model
    if best_preds is not None:
        # Find class names if available
        if hasattr(models_results[0], 'get') and models_results[0].get('label_encoder') is not None:
            class_names = models_results[0]['label_encoder'].classes_
        else:
            class_names = [f"Class {i}" for i in range(len(np.unique(y_test)))]

        # Model names for title
        model_names = [model['name'] for model in models_results]

        print("\nGenerating Optimized Ensemble confusion matrix...")
        fig, cm = plot_confusion_matrix(
            y_test, best_preds,
            class_names=class_names,
            title=f'Optimized {voting_type.capitalize()} Voting Confusion Matrix\n({", ".join(model_names)})',
            filename=f'optimized_{voting_type}_voting_confusion_matrix.png',
            normalize=True
        )

        # Add confusion matrix to result
        best_result['confusion_matrix'] = cm

    return best_result


def run_full_ensemble_experiment(X_train, X_val, X_test, y_train, y_val, y_test, label_encoder):
    """Run the full ensemble experiment with all models and voting strategies."""
    print("\n" + "=" * 50)
    print("Starting Full Ensemble Experiment")
    print("=" * 50)

    # Train individual models
    all_results = []

    # Train traditional models
    svm_results = train_svm_wrapper(X_train, X_val, X_test, y_train, y_val, y_test, label_encoder)
    nb_results = train_nb_wrapper(X_train, X_val, X_test, y_train, y_val, y_test, label_encoder)
    rf_results = train_rf_wrapper(X_train, X_val, X_test, y_train, y_val, y_test, label_encoder)

    # Train deep learning models
    cnn_results = train_cnn_wrapper(X_train, X_val, X_test, y_train, y_val, y_test, label_encoder)

    # Collect all individual models
    individual_models = [svm_results, nb_results, rf_results, cnn_results]

    # Get class names for confusion matrices
    if label_encoder is not None:
        class_names = label_encoder.classes_
    else:
        class_names = [f"Class {i}" for i in range(len(np.unique(y_test)))]

    # Generate confusion matrices for all individual models
    print("\nGenerating confusion matrices for all individual models...")
    for model in individual_models:
        model_name = model['name']
        print(f"\nCreating confusion matrix for {model_name}...")

        # Create confusion matrix
        fig, cm = plot_confusion_matrix(
            y_test, model['predictions'],
            class_names=class_names,
            title=f'{model_name} Confusion Matrix',
            filename=f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png'
        )

        # Store confusion matrix in results
        model['confusion_matrix'] = cm

    # Add individual model results to the results list
    for model in individual_models:
        all_results.append({
            'model': model['name'],
            'accuracy': model['accuracy'],
            'precision': model['precision'],
            'recall': model['recall'],
            'f1': model['f1'],
            'training_time': model['training_time'],
            'confusion_matrix': model.get('confusion_matrix')  # Include confusion matrix
        })

    # Test all model combinations with different voting strategies
    ensemble_results = test_all_model_combinations(individual_models, y_test)

    # Add ensemble results to all results
    all_results.extend(ensemble_results)

    # For the best combination, optimize weights
    best_ensemble = max(ensemble_results, key=lambda x: x['accuracy'])
    model_indices = []
    for name in best_ensemble['model_names']:
        for i, model in enumerate(individual_models):
            if model['name'] == name:
                model_indices.append(i)

    best_models = [individual_models[i] for i in model_indices]

    print("\n" + "=" * 50)
    print(f"Optimizing weights for best model combination: {best_ensemble['model_names']}")

    # Modify optimize_ensemble_weights function to return predictions
    def get_ensemble_predictions(models_results, weights):
        """Get predictions from ensemble with given weights"""
        probabilities = [model['probabilities'] for model in models_results]
        weighted_probs = np.zeros_like(probabilities[0])
        for probs, weight in zip(probabilities, weights):
            weighted_probs += weight * probs
        return np.argmax(weighted_probs, axis=1)

    # Optimize soft voting for the best combination
    optimized_soft = optimize_ensemble_weights(best_models, y_test, 'soft', n_trials=50)

    if optimized_soft:
        optimized_soft['model'] = "Optimized Soft Voting"

        # Get predictions for confusion matrix
        optimized_preds = get_ensemble_predictions(best_models, optimized_soft['weights'])

        # Generate confusion matrix for optimized ensemble
        print("\nGenerating Optimized Ensemble confusion matrix...")
        opt_fig, opt_cm = plot_confusion_matrix(
            y_test, optimized_preds,
            class_names=class_names,
            title='Optimized Soft Voting Ensemble Confusion Matrix',
            filename='optimized_ensemble_confusion_matrix.png',
            normalize=True  # Explicitly set normalization
        )

        # Store confusion matrix and predictions
        optimized_soft['confusion_matrix'] = opt_cm
        optimized_soft['predictions'] = optimized_preds

        all_results.append(optimized_soft)

    # Find overall best model
    best_result = max(all_results, key=lambda x: x['accuracy'])

    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(f"Best Model: {best_result['model']}")
    print(f"Best Accuracy: {best_result['accuracy']:.4f}")
    print(f"Best Precision: {best_result['precision']:.4f}")
    print(f"Best Recall: {best_result['recall']:.4f}")
    print(f"Best F1 Score: {best_result['f1']:.4f}")

    # Create visualization
    create_model_comparison_chart(all_results)

    return all_results


X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = load_data_from_separate_folders()
all_results = run_full_ensemble_experiment(X_train, X_val, X_test, y_train, y_val, y_test, label_encoder)
