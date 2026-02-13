
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

try:
    from class_names import POKEMON_NAMES
except ImportError:
    from src.class_names import POKEMON_NAMES

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 150
DATA_DIR = 'dataset'

def get_class_label(class_idx):
    # class_idx is 0-indexed integer from model output
    # Our folders are '1', '2'... so class_idx 0 -> folder '1' -> Pokemon ID 1
    # Check creating of dataset. class_names are sorted numerically: 1, 10, 100... default sort?
    # NO, we specified class_names explicitely in train_ds!
    # class_names = [str(i) for i in range(1, NUM_CLASSES + 1)]
    # So class_idx 0 corresponds to class_names[0] which is "1" -> Pokemon ID 1.
    pokemon_id = class_idx + 1
    return f"{pokemon_id}. {POKEMON_NAMES.get(pokemon_id, 'Unknown')}"

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.grid(True)
    plt.savefig('training_history.png')
    plt.close()

def plot_confusion_matrix_heatmap(cm, class_names):
    # For 150 classes, annotations are impossible.
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=False, yticklabels=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (150 Classes)')
    plt.savefig('confusion_matrix.png')
    plt.close()

def plot_top_confusions(y_true, y_pred, class_names, top_k=10):
    # Find incorrect predictions
    incorrect_indices = np.where(y_true != y_pred)[0]
    confusions = []
    
    for i in incorrect_indices:
        true_label = get_class_label(y_true[i])
        pred_label = get_class_label(y_pred[i])
        confusions.append(f"{true_label} -> {pred_label}")
    
    if not confusions:
        print("No errors found!")
        return

    from collections import Counter
    confusion_counts = Counter(confusions).most_common(top_k)
    
    pairs = [x[0] for x in confusion_counts]
    counts = [x[1] for x in confusion_counts]
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x=counts, y=pairs, palette='viridis')
    plt.title(f'Top {top_k} Confused Pairs')
    plt.xlabel('Count')
    plt.tight_layout()
    plt.savefig('top_confusions.png')
    plt.close()

def plot_class_performance(y_true, y_pred, class_names):
    # Calculate accuracy per class
    cm = confusion_matrix(y_true, y_pred)
    # cm[i, i] is correct predictions for class i
    # sum(cm[i, :]) is total support for class i
    
    class_accuracies = []
    for i in range(len(class_names)):
        correct = cm[i, i]
        total = np.sum(cm[i, :])
        acc = correct / total if total > 0 else 0
        class_accuracies.append((acc, i))
    
    # Sort by accuracy
    class_accuracies.sort(key=lambda x: x[0])
    
    # Worst 10
    worst_10 = class_accuracies[:10]
    best_10 = class_accuracies[-10:]
    
    # Prepare data for plotting
    worst_labels = [get_class_label(x[1]) for x in worst_10]
    worst_values = [x[0] for x in worst_10]
    
    best_labels = [get_class_label(x[1]) for x in best_10]
    best_values = [x[0] for x in best_10]
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.barplot(x=worst_values, y=worst_labels, palette='Reds_r')
    plt.title('Worst 10 Classes (Accuracy)')
    plt.xlim(0, 1.0)
    
    plt.subplot(1, 2, 2)
    sns.barplot(x=best_values, y=best_labels, palette='Greens_r')
    plt.title('Best 10 Classes (Accuracy)')
    plt.xlim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig('class_performance.png')
    plt.close()

def plot_predictions(model, val_ds, class_names, num_images=16):
    # Get a batch
    images, labels = next(iter(val_ds))
    preds = model.predict(images)
    pred_labels = np.argmax(preds, axis=1)
    
    plt.figure(figsize=(15, 15))
    for i in range(min(num_images, len(images))):
        plt.subplot(4, 4, i + 1)
        img = images[i].numpy().astype("uint8")
        plt.imshow(img)
        
        true_idx = labels[i].numpy() if hasattr(labels[i], 'numpy') else labels[i]
        pred_idx = pred_labels[i]
        
        true_text = get_class_label(true_idx)
        pred_text = get_class_label(pred_idx)
        
        color = 'green' if true_idx == pred_idx else 'red'
        
        plt.title(f"True: {true_text}\nPred: {pred_text}", color=color, fontsize=9)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.close()

def train_model():
    print("Loading dataset...")
    
    # Ensure class names are sorted numerically 1..150
    class_names = [str(i) for i in range(1, NUM_CLASSES + 1)]
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, 'train'),
        labels='inferred',
        label_mode='int',
        class_names=class_names,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, 'val'),
        labels='inferred',
        label_mode='int',
        class_names=class_names, # Use same order
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    # Prefetch
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds_pre = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    print("Building model...")
    base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
    base_model.trainable = False # Freeze base model initially (Transfer Learning)
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x) # Common practice
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss=SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    
    model.summary()
    
    # Check if model exists to skip training? User wants graphs.
    if os.path.exists('pokemon_model.h5'):
        print("Loading existing model...")
        model.load_weights('pokemon_model.h5')
        # We still need history for the plot? 
        # If model is loaded, we can't plot new training history unless we saved it.
        # But we can still generate new evaluation plots.
        # Let's assume re-training is fine or preferred to get history.
        # ACTUALLY, if I just want to generate graphs quickly without retraining if possible...
        # The user said "create understandable graphs".
        # Let's retraining for now to be safe and simple.
    
    print("Training model...")
    history = model.fit(train_ds, validation_data=val_ds_pre, epochs=EPOCHS)
    
    plot_history(history)
    model.save('pokemon_model.h5')
    
    # Evaluation
    print("Evaluating model...")
    y_true = []
    y_pred = []
    
    # Need to iterate over dataset to get all labels
    # Note: val_ds is batched.
    print("Predicting on validation set...")
    for images, labels in val_ds: # Use original val_ds to avoid prefetch issues if any
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix_heatmap(cm, class_names)
    plot_top_confusions(y_true, y_pred, class_names)
    plot_class_performance(y_true, y_pred, class_names)
    plot_predictions(model, val_ds, class_names)
    
    print("Graphical reports generated: training_history.png, confusion_matrix.png, top_confusions.png, class_performance.png, predictions.png")


if __name__ == "__main__":
    train_model()
