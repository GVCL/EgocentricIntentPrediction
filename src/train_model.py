import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier
import pickle
import os

# Define File Paths
data_dir = "emoji-hero-vr-db-sfea-as-csv"
train_file = os.path.join(data_dir, "training_set.csv")
val_file = os.path.join(data_dir, "validation_set.csv")
test_file = os.path.join(data_dir, "test_set.csv")

# Helper function to load data and extract feature names
def load_and_prep(filepath, encoder=None):
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath)
    label_mapping = {
        0: 'Hostile', 1: 'Hostile', 2: 'Fear', 3: 'Happiness', 
        4: 'Neutral', 5: 'Sadness', 6: 'Surprise',
        'Anger': 'Hostile', 'Disgust': 'Hostile'
    }
    df['Label'] = df['Label'].replace(label_mapping)
    
    X_df = df.drop(columns=['file_id', 'timestamp', 'Label'])
    feature_names = X_df.columns.tolist()
    X = X_df.values
    
    y_text = df['Label']
    
    if encoder is None:
        encoder = LabelEncoder()
        y = encoder.fit_transform(y_text)
    else:
        y = encoder.transform(y_text)
        
    return X, y, encoder, feature_names

def main():
    print("--- Preparing Data ---")
    X_train, y_train, label_encoder, feature_names = load_and_prep(train_file)
    X_val, y_val, _, _ = load_and_prep(val_file, encoder=label_encoder)
    X_test, y_test, _, _ = load_and_prep(test_file, encoder=label_encoder)

    # Global Feature Exploration
    print("\n--- Phase 1A: Finding Global Top Features ---")
    global_explorer = XGBClassifier(random_state=42, n_jobs=-1)
    global_explorer.fit(X_train, y_train)
    
    global_importances = pd.DataFrame({'Feature': feature_names, 'Importance': global_explorer.feature_importances_})
    global_top_20 = global_importances.sort_values(by='Importance', ascending=False).head(20)['Feature'].tolist()

    # Targeted Exploration (Fear vs. Surprise)
    print("--- Phase 1B: Finding Silver Bullets for Fear vs. Surprise ---")
    # Find the numeric IDs for Fear and Surprise
    fear_id = label_encoder.transform(['Fear'])[0]
    surprise_id = label_encoder.transform(['Surprise'])[0]
    
    # Isolate only Fear and Surprise data
    mask = (y_train == fear_id) | (y_train == surprise_id)
    X_train_fs = X_train[mask]
    y_train_fs = y_train[mask]
    
    # Temporarily convert to 0 and 1 for binary classification
    y_train_fs_binary = (y_train_fs == surprise_id).astype(int)
    
    fs_explorer = XGBClassifier(random_state=42, n_jobs=-1)
    fs_explorer.fit(X_train_fs, y_train_fs_binary)
    
    fs_importances = pd.DataFrame({'Feature': feature_names, 'Importance': fs_explorer.feature_importances_})
    fs_top_10 = fs_importances.sort_values(by='Importance', ascending=False).head(10)['Feature'].tolist()

    # Combining Feature Sets
    combined_features = list(set(global_top_20 + fs_top_10))
    print(f"\nKeeping {len(combined_features)} unique features optimized for overall accuracy AND Fear/Surprise distinction.")
    
    top_indices = [feature_names.index(f) for f in combined_features]

    # Retraining on Clean Data
    print("\n--- Phase 2: Retraining Model on Customized Features ---")
    X_train_clean = X_train[:, top_indices]
    X_val_clean = X_val[:, top_indices]
    X_test_clean = X_test[:, top_indices]

    final_model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss',
        early_stopping_rounds=25
    )

    final_model.fit(
        X_train_clean, y_train,
        eval_set=[(X_val_clean, y_val)],
        verbose=False
    )
    
    print(f"Training stopped at tree {final_model.best_iteration}.")

    # Evaluate
    print("\n--- Validation Set Results ---")
    y_val_pred = final_model.predict(X_val_clean)
    print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred) * 100:.2f}%")

    print("\n--- Test Set Results ---")
    y_test_pred = final_model.predict(X_test_clean)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    print("\nDetailed Classification Report (Test Set):")
    target_names_str = [str(c) for c in label_encoder.classes_]
    print(classification_report(y_test, y_test_pred, target_names=target_names_str))

    # Save Artifacts
    print("\n--- Generating Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Set up the figure with a slightly taller ratio to accommodate vertical labels
    plt.figure(figsize=(10, 8))
    
    # annot_kws={"size": 14} makes the numbers inside the boxes much larger
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names_str, yticklabels=target_names_str,
                annot_kws={"size": 14})
    
    # Force X labels to be vertical (90 degrees) and Y labels to be horizontal (0 degrees)
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    
    # Make axis labels bold and larger
    plt.ylabel('Actual', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted', fontsize=14, fontweight='bold')
    plt.title('Customized Features Emotion Classification', fontsize=16, pad=20)
    
    # tight_layout prevents the vertical labels from getting cut off at the bottom
    plt.tight_layout()
    
    # Save at 300 DPI (dots per inch) - this is mandatory for crisp text in academic papers
    plt.savefig('confusion_matrix_custom.png', dpi=300, bbox_inches='tight')
    
    print("\n--- Saving Artifacts ---")
    with open('xgb_custom_model.pkl', 'wb') as f:
        pickle.dump(final_model, f)
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    with open('top_feature_indices.pkl', 'wb') as f:
        pickle.dump(top_indices, f)

    print("Pipeline complete! Artifacts saved.")

if __name__ == "__main__":
    main()