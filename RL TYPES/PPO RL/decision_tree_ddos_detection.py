# Import necessary libraries
try:
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    print(f"Error: {e}")
    print("Please install the required libraries using:")
    print("pip install pandas numpy scikit-learn matplotlib seaborn")
    exit()

# Step 1: Load and Preprocess the Dataset
def load_and_preprocess_data(file_path, target_column='Label'):
    """
    Load and preprocess the dataset.
    """
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Strip whitespace from column names (in case there are extra spaces)
    data.columns = data.columns.str.strip()
    
    # Replace infinite values with NaN and drop them
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    
    # Dynamically identify the target column
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset. Available columns: {data.columns}")
    
    # Dynamically identify feature columns (exclude non-numeric and target columns)
    non_feature_columns = [target_column, 'Flow ID', 'Source IP', 'Destination IP', 'Timestamp']  # Exclude non-numeric columns
    feature_columns = [col for col in data.columns if col not in non_feature_columns and np.issubdtype(data[col].dtype, np.number)]
    
    # Extract features and labels
    X = data[feature_columns]
    y = data[target_column]  # Use the specified target column
    
    # Encode labels (e.g., 'Benign' -> 0, 'DDoS' -> 1)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, feature_columns

# Step 2: Train the Decision Tree Model
def train_decision_tree(X_train, y_train):
    """
    Train a Decision Tree classifier.
    """
    # Initialize the Decision Tree model
    model = DecisionTreeClassifier(random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    return model

# Step 3: Evaluate the Model
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and print performance metrics.
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Print metrics
    print("Model Performance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Plot Confusion Matrix
    plot_confusion_matrix(y_test, y_pred)

# Step 4: Plot Confusion Matrix
def plot_confusion_matrix(y_true, y_pred):
    """
    Plot a confusion matrix for the model's predictions.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Benign', 'DDoS'],
                yticklabels=['Benign', 'DDoS'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Main Function
if __name__ == "__main__":
    # Load and preprocess the data
    file_path = 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'  # Replace with the actual path to the dataset
    target_column = 'Label'  # Replace with the correct target column name in your dataset
    try:
        X_train, X_test, y_train, y_test, feature_columns = load_and_preprocess_data(file_path, target_column)
        
        # Print the detected feature columns
        print("Detected Feature Columns:", feature_columns)
        
        # Train the Decision Tree model
        model = train_decision_tree(X_train, y_train)
        
        # Evaluate the model
        evaluate_model(model, X_test, y_test)
    except ValueError as e:
        print(e)