import os
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score
from .preprocess import load_dataset
from .model import create_model
import numpy as np

def train_model(train_dir, test_dir, model_save_path, graphs_dir):
    print("Loading training data...")
    X_train, y_train = load_dataset(train_dir)
    print("Loading testing data...")
    X_test, y_test = load_dataset(test_dir)
    
    if len(X_train) == 0 or len(X_test) == 0:
        return None, "Dataset is empty or path is incorrect."
    model = create_model()
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=32
    )
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    accuracy = history.history['accuracy'][-1]
    loss = history.history['loss'][-1]
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    os.makedirs(graphs_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(graphs_dir, 'accuracy_plot.png'))
    plt.close()
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(graphs_dir, 'loss_plot.png'))
    plt.close()
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['FAKE', 'REAL'], yticklabels=['FAKE', 'REAL'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(graphs_dir, 'confusion_matrix.png'))
    plt.close()
    
    metrics = {
        'accuracy': round(accuracy, 4),
        'loss': round(loss, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4)
    }
    
    return metrics, None
