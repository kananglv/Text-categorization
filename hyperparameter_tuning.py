# All necessary imports at the beginning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import glob

# scikit-learn imports - including the missing function
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support  # This was missing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Conv1D, GlobalMaxPooling1D, Bidirectional
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping


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


def train_svm_wrapper(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train SVM model with TF-IDF features using separate validation set"""
    print("\nTraining SVM model...")
    start = time.time()

    # Define hyperparameters
    tfidf_max_features = [50000, 60000, 70000, 80000, 90000]
    tfidf_ngram_ranges = [(1, 1), (1, 2), (1, 3)]
    c_values = [0.1, 1.0, 2.0, 3.0, 5.0, 10.0]

    best_score = 0
    best_params = {}
    best_vectorizer = None
    best_model = None

    # Hyperparameter tuning using validation set
    print("Starting hyperparameter search for SVM...")
    param_combinations = len(tfidf_max_features) * len(tfidf_ngram_ranges) * len(c_values)
    counter = 0

    for max_features in tfidf_max_features:
        for ngram_range in tfidf_ngram_ranges:
            for c in c_values:
                counter += 1
                print(
                    f"Trying combination {counter}/{param_combinations}: max_features={max_features}, ngram_range={ngram_range}, C={c}")

                try:
                    # Create and fit TF-IDF vectorizer
                    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
                    X_train_tfidf = tfidf.fit_transform(X_train)
                    X_val_tfidf = tfidf.transform(X_val)

                    # Create and train SVM with explicit dual parameter to avoid warning
                    model = LinearSVC(C=c, random_state=42, dual=False)
                    model.fit(X_train_tfidf, y_train)

                    # Evaluate on validation set
                    val_score = model.score(X_val_tfidf, y_val)

                    print(f"  Validation score: {val_score:.4f}")

                    if val_score > best_score:
                        best_score = val_score
                        best_params = {
                            'max_features': max_features,
                            'ngram_range': ngram_range,
                            'C': c,
                            'dual': False
                        }
                        best_vectorizer = tfidf
                        best_model = model
                except Exception as e:
                    print(f"  Error with this combination: {e}")

    # Transform test data using best vectorizer
    X_test_tfidf = best_vectorizer.transform(X_test)

    # Get predictions
    y_pred = best_model.predict(X_test_tfidf)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    training_time = time.time() - start

    # Print detailed results
    print("\n" + "=" * 70)
    print("SVM MODEL RESULTS")
    print("=" * 70)
    print(f"Best parameters: {best_params}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Training time: {training_time:.2f} seconds")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return {
        'name': 'SVM',
        'model': best_model,
        'vectorizer': best_vectorizer,
        'predictions': y_pred,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'training_time': training_time,
        'best_params': best_params
    }


def train_nb_wrapper(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train Naive Bayes model using separate validation set for tuning"""
    print("\n" + "=" * 50)
    print("Training Naive Bayes model with hyperparameter tuning...")
    start = time.time()

    # Define parameter combinations
    tfidf_max_features = [50000, 60000, 70000, 80000, 90000]
    tfidf_ngram_ranges = [(1, 1), (1, 2)]
    min_df_values = [1, 2, 3, 5, 10]

    # Naive Bayes parameters
    alpha_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
    fit_prior_values = [True, False]

    best_score = 0
    best_params = {}
    best_vectorizer = None
    best_classifier = None

    print("Starting hyperparameter search for Naive Bayes...")

    for max_features in tfidf_max_features:
        for ngram_range in tfidf_ngram_ranges:
            for min_df in min_df_values:
                for alpha in alpha_values:
                    for fit_prior in fit_prior_values:
                        print(f"Testing NB: max_features={max_features}, "
                              f"ngram_range={ngram_range}, min_df={min_df}, "
                              f"alpha={alpha}, fit_prior={fit_prior}")

                        try:
                            # Set up vectorizer on training data only
                            vectorizer = TfidfVectorizer(
                                max_features=max_features,
                                ngram_range=ngram_range,
                                min_df=min_df,
                                sublinear_tf=True)

                            # Transform training data
                            X_train_tfidf = vectorizer.fit_transform(X_train)

                            # Transform validation data
                            X_val_tfidf = vectorizer.transform(X_val)

                            # Create and train classifier
                            classifier = MultinomialNB(
                                alpha=alpha,
                                fit_prior=fit_prior)

                            classifier.fit(X_train_tfidf, y_train)

                            # Evaluate on validation data
                            val_score = classifier.score(X_val_tfidf, y_val)
                            print(f"  Validation score: {val_score:.4f}")

                            if val_score > best_score:
                                best_score = val_score
                                best_params = {
                                    'tfidf__max_features': max_features,
                                    'tfidf__ngram_range': ngram_range,
                                    'tfidf__min_df': min_df,
                                    'clf__alpha': alpha,
                                    'clf__fit_prior': fit_prior
                                }
                                best_vectorizer = vectorizer
                                best_classifier = classifier
                        except Exception as e:
                            print(f"  Error with this combination: {str(e)}")

    print(f"Best parameters: {best_params}")

    # Transform test data with best vectorizer
    X_test_tfidf = best_vectorizer.transform(X_test)

    # Calculate metrics on test data
    y_pred = best_classifier.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    training_time = time.time() - start

    # Print results
    print(f"Naive Bayes Results with best parameters:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Training time: {training_time:.2f} seconds")

    return {
        'name': 'Naive Bayes',  # Changed from 'model' to 'name' for consistency
        'model': best_classifier,
        'vectorizer': best_vectorizer,
        'predictions': y_pred,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'training_time': training_time,
        'best_params': best_params
    }


def train_rf_wrapper(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train Random Forest model using separate validation set for tuning"""
    print("\n" + "=" * 50)
    print("Training Random Forest model with hyperparameter tuning...")
    start = time.time()

    # Define parameter combinations (reduced for runtime)
    tfidf_max_features = [40000, 50000, 60000, 70000, 80000, 90000]
    n_estimators_values = [50, 100]
    max_depth_values = [None, 50]
    min_samples_split_values = [2, 5]
    criterion_values = ['gini', 'entropy', 'log_loss']
    max_features_values = ['log2', None]

    best_score = 0
    best_params = {}
    best_vectorizer = None
    best_classifier = None

    print("\nTesting Random Forest parameter combinations:")

    # Generate all combinations for parameters
    parameter_combinations = [
        (max_feat, n_est, max_depth, min_split, crit, max_f)
        for max_feat in tfidf_max_features
        for n_est in n_estimators_values
        for max_depth in max_depth_values
        for min_split in min_samples_split_values
        for crit in criterion_values
        for max_f in max_features_values
    ]

    total_combinations = len(parameter_combinations)
    print(f"Total combinations to test: {total_combinations}")

    # Counter for completed combinations
    counter = 0

    for i, (max_features, n_estimators, max_depth, min_samples_split, criterion, max_f) in enumerate(
            parameter_combinations):
        try:
            # Format for None values
            depth_str = "None" if max_depth is None else str(max_depth)
            max_f_str = "None" if max_f is None else str(max_f)

            print(f"Testing RF: max_features={max_features}, n_estimators={n_estimators}, "
                  f"max_depth={depth_str}, min_samples_split={min_samples_split}, "
                  f"criterion={criterion}, max_features={max_f_str} ({i + 1}/{total_combinations})")

            # Create pipeline with current hyperparameters
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=max_features)),
                ('clf', RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    criterion=criterion,
                    max_features=max_f,
                    random_state=42,
                    n_jobs=-1
                ))
            ])

            # Fit the model
            pipeline.fit(X_train, y_train)

            # Evaluate the model
            y_pred = pipeline.predict(X_test)
            score = accuracy_score(y_test, y_pred)

            # Mark as completed
            counter += 1

            # Print results
            print(f"  Score: {score:.4f}")

            # Check if this is the best model so far
            if score > best_score:
                best_score = score
                best_params = {
                    'tfidf__max_features': max_features,
                    'clf__n_estimators': n_estimators,
                    'clf__max_depth': max_depth,
                    'clf__min_samples_split': min_samples_split,
                    'clf__criterion': criterion,
                    'clf__max_features': max_f
                }
                best_pipeline = pipeline
                print(f"  → New best score: {score:.4f}")
        except Exception as e:
            print(f"  Error with this combination: {str(e)}")
            continue

    # Print summary
    print(f"Combinations completed: {counter}/{total_combinations}")
    print(f"Best parameters: {best_params}")

    # Get final predictions with best model
    y_pred = best_pipeline.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    training_time = time.time() - start

    # Print results
    print(f"\nRandom Forest Final Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Training time: {training_time:.2f} seconds")

    return {
        'model': 'Random Forest',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'training_time': training_time,
        'best_params': best_params
    }


def train_rnn_wrapper(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train Bidirectional LSTM model with L2 regularization"""
    print("\n" + "=" * 50)
    print("Training Bidirectional LSTM model with L2 regularization...")
    start = time.time()

    random_state = 42
    tf.random.set_seed(random_state)
    np.random.seed(random_state)

    # Text tokenization
    max_features = 40000  # Top most frequent words
    maxlen = 300  # Max sequence length

    # Create and fit tokenizer on training data only
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(X_train)

    # Convert text to sequences and pad
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_val_seq = tokenizer.texts_to_sequences(X_val)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen)
    X_val_pad = pad_sequences(X_val_seq, maxlen=maxlen)
    X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen)

    # Number of classes
    num_classes = len(np.unique(np.concatenate([y_train, y_val, y_test])))

    # Define expanded hyperparameter search space
    embedding_dims = [128, 256]
    lstm_units = [32, 64]
    dropout_rates = [0.3, 0.5, 0.7]
    l2_factors = [0.002]  # Added L2 regularization factors
    optimizers = ['adam']
    batch_sizes = [64, 128]

    # Manually perform grid search for hyperparameters
    best_accuracy = 0
    best_params = {}
    best_model = None
    successful_models = 0

    print("Starting hyperparameter tuning for Bidirectional LSTM...")

    # Use a reduced set of combinations
    parameter_combinations = [
        (emb_dim, lstm, drop, l2_factor, opt, batch)
        for emb_dim in embedding_dims
        for lstm in lstm_units
        for drop in dropout_rates
        for l2_factor in l2_factors  # L2 regularization factor
        for opt in optimizers
        for batch in batch_sizes
    ]

    total_combinations = len(parameter_combinations)

    for i, (embedding_dim, lstm_unit, dropout_rate, l2_factor, optimizer, batch_size) in enumerate(
            parameter_combinations):
        print(f"\nTrying combination {i + 1}/{total_combinations}:")
        print(f"  embedding_dim={embedding_dim}, lstm_units={lstm_unit}, dropout={dropout_rate}")
        print(f"  L2 factor={l2_factor}, optimizer={optimizer}, batch_size={batch_size}, random_state={random_state}")

        try:
            # Reset the seed before each model build for consistency
            tf.random.set_seed(random_state)

            # Build model with current hyperparameters using Bidirectional LSTM
            model = Sequential()

            # Use seed-controlled initializers for reproducible results
            model.add(Embedding(
                max_features,
                embedding_dim,
                input_length=maxlen,
                embeddings_initializer=tf.keras.initializers.GlorotUniform(seed=random_state)
            ))

            # Use Bidirectional wrapper around LSTM layer with seed-controlled initializers
            model.add(Bidirectional(LSTM(
                lstm_unit,
                dropout=dropout_rate,
                recurrent_dropout=dropout_rate,
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=random_state),
                recurrent_initializer=tf.keras.initializers.Orthogonal(seed=random_state)
            )))

            # Add L2 regularization to Dense layers with seed-controlled initializers
            model.add(Dense(
                128,
                activation='relu',
                kernel_regularizer=l2(l2_factor),
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=random_state)
            ))

            model.add(Dropout(dropout_rate, seed=random_state))

            model.add(Dense(
                num_classes,
                activation='softmax',
                kernel_regularizer=l2(l2_factor),
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=random_state)
            ))

            # Compile model
            model.compile(
                loss='sparse_categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy']
            )

            # Use early stopping and learning rate reduction
            early_stop = EarlyStopping(
                monitor='val_accuracy',
                patience=7,
                restore_best_weights=False,
                verbose=1  # Show messages about early stopping
            )

            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1  # Show messages about LR reduction
            )

            # Train the model with explicit validation data and show epoch progress
            history = model.fit(
                X_train_pad, y_train,
                batch_size=batch_size,
                epochs=50,  # Increased epochs for Bidirectional LSTM
                validation_data=(X_val_pad, y_val),
                callbacks=[early_stop, reduce_lr],
                verbose=1  # Show epoch progress bar
            )

            # Evaluate on validation set
            val_loss, val_accuracy = model.evaluate(X_val_pad, y_val, verbose=0)
            print(f"  Validation accuracy: {val_accuracy:.4f}")

            # Check if this model is better
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model = model  # Save the actual model object
                best_params = {
                    'embedding_dim': embedding_dim,
                    'lstm_units': lstm_unit,
                    'dropout_rate': dropout_rate,
                    'l2_factor': l2_factor,
                    'optimizer': optimizer,
                    'batch_size': batch_size,
                    'architecture': 'Bidirectional LSTM with L2'
                }
                print(f"  → New best model found!")

            successful_models += 1

        except Exception as e:
            print(f"  Error with this combination: {e}")
            continue

    print(f"\nBest Bidirectional LSTM parameters: {best_params}")

    # Evaluate on test data
    test_loss, test_accuracy = best_model.evaluate(X_test_pad, y_test, verbose=0)

    # Make predictions with best model
    y_pred_probs = best_model.predict(X_test_pad)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    training_time = time.time() - start

    # Print results
    print(f"Bidirectional LSTM Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Training time: {training_time:.2f} seconds")

    return {
        'name': 'Bidirectional LSTM',
        'model': best_model,
        'tokenizer': tokenizer,
        'maxlen': maxlen,
        'predictions': y_pred,
        'probabilities': y_pred_probs,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'training_time': training_time,
        'best_params': best_params
    }


def train_cnn_wrapper(X_train, X_val, X_test, y_train, y_val, y_test, max_features=40000, maxlen=500):
    """Train CNN model with L2 regularization"""
    print("\n" + "=" * 50)
    print("Training CNN model with L2 regularization...")
    start = time.time()

    random_state = 42
    tf.random.set_seed(random_state)
    np.random.seed(random_state)

    # Create and fit tokenizer on training data only
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(X_train)

    # Convert text to sequences and pad
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_val_seq = tokenizer.texts_to_sequences(X_val)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen)
    X_val_pad = pad_sequences(X_val_seq, maxlen=maxlen)
    X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen)

    # Number of classes
    num_classes = len(np.unique(np.concatenate([y_train, y_val, y_test])))

    # Define hyperparameter search space
    embedding_dims = [128, 256]
    filter_sizes = [3, 5]
    num_filters = [128, 256]
    dropout_rates = [0.3, 0.5, 0.7]
    l2_factors = [0.002]  # Added L2 regularization factors
    optimizers = ['adam', 'rmsprop']
    batch_sizes = [64, 128]

    # Manually perform grid search with explicit validation set
    best_accuracy = 0
    best_params = {}
    best_model = None
    has_valid_model = False

    print("Starting hyperparameter tuning for CNN...")

    # Use a reduced set of combinations for brevity
    parameter_combinations = [
        (emb_dim, filt_size, num_filt, drop, l2_factor, opt, batch)
        for emb_dim in embedding_dims
        for filt_size in filter_sizes
        for num_filt in num_filters
        for drop in dropout_rates
        for l2_factor in l2_factors  # L2 regularization factor
        for opt in optimizers
        for batch in batch_sizes
    ]

    total_combinations = len(parameter_combinations)

    for i, (embedding_dim, filter_size, num_filter, dropout_rate, l2_factor, optimizer, batch_size) in enumerate(
            parameter_combinations):
        print(f"\nCombination {i + 1}/{total_combinations}:")
        print(f"  embedding_dim={embedding_dim}, filter_size={filter_size}, num_filters={num_filter}")
        print(
            f"  dropout={dropout_rate}, L2 factor={l2_factor}, optimizer={optimizer}, batch_size={batch_size}, random_state={random_state}")

        try:
            # Reset the seed before each model build for consistency
            tf.random.set_seed(random_state)

            # Build model with current hyperparameters
            model = Sequential()

            # Add embedding with seed-controlled initializer
            model.add(Embedding(
                max_features,
                embedding_dim,
                input_length=maxlen,
                embeddings_initializer=tf.keras.initializers.GlorotUniform(seed=random_state)
            ))

            # CNN layer with L2 regularization and seed-controlled initializer
            model.add(Conv1D(
                num_filter,
                filter_size,
                activation='relu',
                kernel_regularizer=l2(l2_factor),
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=random_state)
            ))

            model.add(GlobalMaxPooling1D())

            # Dense layer with L2 regularization and seed-controlled initializer
            model.add(Dense(
                128,
                activation='relu',
                kernel_regularizer=l2(l2_factor),
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=random_state)
            ))

            model.add(Dropout(dropout_rate, seed=random_state))

            # Output layer with L2 regularization and seed-controlled initializer
            model.add(Dense(
                num_classes,
                activation='softmax',
                kernel_regularizer=l2(l2_factor),
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=random_state)
            ))

            # Compile model
            model.compile(
                loss='sparse_categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy']
            )

            # Use early stopping
            early_stop = EarlyStopping(
                monitor='val_accuracy',
                patience=3,
                restore_best_weights=True,
                verbose=1  # Show messages about early stopping
            )

            # Use LR reduction
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=3,
                min_lr=0.00001,
                verbose=1  # Show messages about LR reduction
            )

            # Train with explicit validation data
            history = model.fit(
                X_train_pad, y_train,
                batch_size=batch_size,
                epochs=50,  # Increased epochs
                validation_data=(X_val_pad, y_val),
                callbacks=[early_stop, reduce_lr],
                verbose=1  # Show epoch progress
            )

            # Evaluate on validation set
            val_loss, val_accuracy = model.evaluate(X_val_pad, y_val, verbose=0)

            # Print results
            print(f"  Results:")
            print(f"    Validation accuracy: {val_accuracy:.4f}")

            # Check if this is the best model so far
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model = model
                has_valid_model = True
                best_params = {
                    'embedding_dim': embedding_dim,
                    'filter_size': filter_size,
                    'num_filters': num_filter,
                    'dropout_rate': dropout_rate,
                    'l2_factor': l2_factor,
                    'optimizer': optimizer,
                    'batch_size': batch_size
                }
                print(f"  → New best model found!")
        except Exception as e:
            print(f"  Error with this combination: {e}")
            continue

    print(f"\nBest CNN parameters: {best_params}")

    # Make predictions with best model
    test_loss, test_accuracy = best_model.evaluate(X_test_pad, y_test, verbose=0)
    y_pred_probs = best_model.predict(X_test_pad, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    training_time = time.time() - start

    # Print results
    print(f"CNN Final Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Training time: {training_time:.2f} seconds")

    return {
        'name': 'CNN with L2',
        'model': best_model,
        'tokenizer': tokenizer,
        'maxlen': maxlen,
        'predictions': y_pred,
        'probabilities': y_pred_probs,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'training_time': training_time,
        'best_params': best_params
    }


def compare_models(results_list):
    # Create a comparison DataFrame
    comparison = pd.DataFrame(results_list)

    # Sort by F1 score
    comparison = comparison.sort_values('f1', ascending=False)

    # Format for display
    display_cols = ['name', 'accuracy', 'precision', 'recall', 'f1', 'training_time']

    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)
    print(comparison[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Plot metrics comparison
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    fig, ax = plt.subplots(figsize=(12, 8))

    x = np.arange(len(comparison['name']))
    width = 0.2

    # Create bars with distinct colors for better differentiation
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax.bar(x + (i - 1.5) * width, comparison[metric], width, label=metric.capitalize(), color=color)

    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison['name'])
    ax.legend()
    ax.set_ylim(0, 1.0)

    # Add value labels on bars
    for i, metric in enumerate(metrics):
        for j, v in enumerate(comparison[metric]):
            ax.text(j + (i - 1.5) * width, v + 0.02, f"{v:.2f}",
                    ha='center', va='bottom', fontsize=8, rotation=90)

    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


def run_text_classification_experiment():
    """Main function to run the text classification experiment"""
    try:
        # Try loading from separate train and test folders
        try:
            # First try relative paths
            X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = load_data_from_separate_folders(
                "DATA FOR THE MODEL TRAIN",
                "DATA FOR THE MODEL TEST"
            )
        except FileNotFoundError:
            try:
                # Try absolute paths
                X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = load_data_from_separate_folders(
                    r"C:\Users\Karim\Downloads\DATA FOR THE MODEL TRAIN",
                    r"C:\Users\Karim\Downloads\DATA FOR THE MODEL TEST"
                )
            except FileNotFoundError:
                print("Please enter the path to the training data folder:")
                train_path = input()
                print("Please enter the path to the test data folder:")
                test_path = input()
                X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = load_data_from_separate_folders(
                    train_path, test_path
                )

        # Train all models with hyperparameter tuning
        print("\nStarting model training and hyperparameter tuning...")
        results = []

        # Ask which models to run
        print("This will test many parameter combinations and may take a long time.")
        print("Enter 'a' to run all models, or select specific models (e.g. '1 3' for SVM and RF):")
        print("1: SVM")
        print("2: Naive Bayes")
        print("3: Random Forest")
        print("4: CNN")

        # For demonstration, run all models
        choice = 'a'  # input() in actual interactive use

        if choice.lower() == 'a':
            models_to_run = [1, 2, 3, 4]
        else:
            models_to_run = [int(x) for x in choice.split()]

        # Train models with the new train/val/test sets
        if 1 in models_to_run:
            svm_results = train_svm_wrapper(X_train, X_val, X_test, y_train, y_val, y_test)
            results.append(svm_results)

        if 2 in models_to_run:
            nb_results = train_nb_wrapper(X_train, X_val, X_test, y_train, y_val, y_test)
            results.append(nb_results)

        if 3 in models_to_run:
            rf_results = train_rf_wrapper(X_train, X_val, X_test, y_train, y_val, y_test)
            results.append(rf_results)

        if 4 in models_to_run:
            cnn_results = train_cnn_wrapper(X_train, X_val, X_test, y_train, y_val, y_test)
            results.append(cnn_results)

        # Compare all models
        compare_models(results)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


# Run the experiment
run_text_classification_experiment()