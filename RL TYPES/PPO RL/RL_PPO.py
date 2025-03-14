# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import gymnasium as gym  # Use Gymnasium instead of OpenAI Gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from sklearn.metrics import classification_report

# Step 1: Load and Preprocess the Dataset
def load_and_preprocess_data(file_path, target_column='Label'):
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

# Step 2: Create a Custom Gymnasium Environment for DDoS Detection
class DDoSEnv(gym.Env):
    def __init__(self, X, y):
        super(DDoSEnv, self).__init__()
        self.X = X
        self.y = y
        self.current_step = 0
        
        # Define action and observation space
        self.action_space = spaces.Discrete(2)  # 0: benign, 1: DDoS
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(X.shape[1],), dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        self.current_step = 0
        return self.X[self.current_step], {}
    
    def step(self, action):
        # Calculate reward
        reward = 1 if action == self.y[self.current_step] else -1
        
        # Move to the next step
        self.current_step += 1
        
        # Check if the episode is done
        done = self.current_step >= len(self.X) - 1
        
        # Return next observation, reward, done, and info
        return self.X[self.current_step], reward, done, False, {}
    
# Step 3: Train the RL Model
def train_rl_model(X_train, y_train):
    # Create the environment
    env = make_vec_env(lambda: DDoSEnv(X_train, y_train), n_envs=1)
    
    # Initialize the PPO model
    model = PPO('MlpPolicy', env, verbose=1)
    
    # Train the model
    model.learn(total_timesteps=10000)
    
    return model

# Step 4: Evaluate the RL Model
def evaluate_model(model, X_test, y_test):
    # Create the test environment
    env = DDoSEnv(X_test, y_test)
    obs, _ = env.reset()
    
    # Run the model on the test data
    predictions = []
    for _ in range(len(X_test)):
        action, _ = model.predict(obs)
        predictions.append(action)
        obs, _, done, _, _ = env.step(action)
        if done:
            break
    
    # Print classification report
    print(classification_report(y_test[:len(predictions)], predictions, target_names=['Benign', 'DDoS']))

# Main Function
if __name__ == "__main__":
    # Load and preprocess the data
    file_path = 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'  # Replace with the actual path to the dataset
    target_column = 'Label'  # Replace with the correct target column name in your dataset
    try:
        X_train, X_test, y_train, y_test, feature_columns = load_and_preprocess_data(file_path, target_column)
        
        # Print the detected feature columns
        print("Detected Feature Columns:", feature_columns)
        
        # Train the RL model
        model = train_rl_model(X_train, y_train)
        
        # Evaluate the model
        evaluate_model(model, X_test, y_test)
    except ValueError as e:
        print(e)