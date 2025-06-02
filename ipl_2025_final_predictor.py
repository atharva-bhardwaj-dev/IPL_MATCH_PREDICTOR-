import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class CorrectedIPLNeuralPredictor:
    def __init__(self, data_path=None, player_data=None):
        """Initialize with data path or DataFrame"""
        if player_data is not None:
            self.df = player_data.copy()
        elif data_path:
            self.df = pd.read_csv(data_path)
        else:
            raise ValueError("Either data_path or player_data must be provided")
            
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.team_stats_cache = {}
        
        # Clean and prepare data
        self._clean_data()
        
    def _clean_data(self):
        """Clean and validate the dataset"""
        print("Cleaning dataset...")
        
        # Handle missing values properly
        numeric_columns = ['runs', 'wickets', 'matches', 'batting_avg', 'batting_strike_rate', 
                          'boundaries_percent', 'bowling_avg', 'bowling_economy', 
                          'bowling_strike_rate', 'catches', 'stumpings']
        
        for col in numeric_columns:
            if col in self.df.columns:
                # Replace unrealistic values
                if col == 'bowling_avg':
                    self.df[col] = self.df[col].replace([np.inf, -np.inf], np.nan)
                    self.df.loc[self.df[col] > 100, col] = np.nan
                elif col == 'bowling_economy':
                    self.df.loc[self.df[col] > 20, col] = np.nan
                elif col == 'bowling_strike_rate':
                    self.df.loc[self.df[col] > 50, col] = np.nan
                    
                # Fill NaN with appropriate values
                if col in ['batting_avg', 'bowling_avg']:
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                else:
                    self.df[col] = self.df[col].fillna(0)
        
        print(f"Dataset cleaned. Shape: {self.df.shape}")
        
    def calculate_normalized_player_score(self, player_data):
        """Calculate normalized player impact score with proper scaling"""
        batting_score = 0
        bowling_score = 0
        fielding_score = 0
        
        # Normalized batting score (0-1 scale)
        if player_data['runs'] > 0:
            # Normalize each component to 0-1 scale
            avg_norm = min(player_data['batting_avg'] / 50, 1)  # Cap at 50
            sr_norm = min((player_data['batting_strike_rate'] - 80) / 100, 1)  # Normalize around 80-180
            boundary_norm = min(player_data['boundaries_percent'] / 30, 1)  # Cap at 30%
            
            batting_score = (avg_norm * 0.4 + max(sr_norm, 0) * 0.4 + boundary_norm * 0.2)
            
        # Normalized bowling score (0-1 scale)
        if player_data['wickets'] > 0:
            # Lower is better for bowling avg and economy
            avg_norm = max(0, (40 - min(player_data['bowling_avg'], 40)) / 40)
            econ_norm = max(0, (12 - min(player_data['bowling_economy'], 12)) / 12)
            sr_norm = max(0, (30 - min(player_data['bowling_strike_rate'], 30)) / 30)
            
            bowling_score = (avg_norm * 0.4 + econ_norm * 0.3 + sr_norm * 0.3)
            
        # Fielding score (normalized by matches)
        if player_data['matches'] > 0:
            fielding_score = min((player_data['catches'] + player_data['stumpings']) / player_data['matches'], 1)
        
        # Experience factor
        experience_factor = min(player_data['matches'] / 100, 1)
        
        # Combined score with experience weighting
        total_score = (batting_score + bowling_score + fielding_score) * (0.7 + 0.3 * experience_factor)
        
        return min(total_score, 1)  # Cap at 1
    
    def calculate_team_features(self, team_players):
        """Calculate comprehensive team features with proper validation"""
        if not team_players:
            return None
            
        # Filter valid players
        valid_players = [p for p in team_players if p in self.df['player'].values]
        if len(valid_players) < 5:
            return None
            
        team_data = self.df[self.df['player'].isin(valid_players)].copy()
        
        # Calculate individual scores
        team_data['player_score'] = team_data.apply(self.calculate_normalized_player_score, axis=1)
        
        # Team composition analysis
        batsmen = team_data[team_data['runs'] >= 500]
        bowlers = team_data[team_data['wickets'] >= 20]
        all_rounders = team_data[(team_data['runs'] >= 300) & (team_data['wickets'] >= 15)]
        experienced = team_data[team_data['matches'] >= 50]
        
        # Safe calculation with proper handling of empty series
        def safe_mean(series, default=0):
            return series.mean() if len(series) > 0 and not series.isna().all() else default
        
        def safe_max(series, default=0):
            return series.max() if len(series) > 0 and not series.isna().all() else default
        
        features = {
            # Batting metrics
            'batting_strength': safe_mean(team_data[team_data['runs'] > 0]['batting_avg']),
            'batting_aggression': safe_mean(team_data[team_data['runs'] > 0]['batting_strike_rate']),
            'boundary_power': safe_mean(team_data[team_data['runs'] > 0]['boundaries_percent']),
            'total_runs': team_data['runs'].sum(),
            'top_scorer': safe_max(team_data['runs']),
            
            # Bowling metrics  
            'bowling_strength': 40 - safe_mean(team_data[team_data['wickets'] > 0]['bowling_avg']),
            'bowling_economy': 10 - safe_mean(team_data[team_data['wickets'] > 0]['bowling_economy']),
            'bowling_strike_rate': 25 - safe_mean(team_data[team_data['wickets'] > 0]['bowling_strike_rate']),
            'total_wickets': team_data['wickets'].sum(),
            'top_bowler': safe_max(team_data['wickets']),
            
            # Team composition
            'num_batsmen': len(batsmen),
            'num_bowlers': len(bowlers),
            'num_all_rounders': len(all_rounders),
            'num_experienced': len(experienced),
            'team_depth': len(team_data),
            
            # Team quality metrics
            'avg_player_score': safe_mean(team_data['player_score']),
            'top_player_score': safe_max(team_data['player_score']),
            'team_experience': safe_mean(team_data['matches']),
            'fielding_strength': team_data['catches'].sum() + team_data['stumpings'].sum(),
            
            # Balance metrics
            'batting_depth': len(team_data[team_data['runs'] >= 200]),
            'bowling_depth': len(team_data[team_data['wickets'] >= 10]),
            'balance_score': len(all_rounders) / max(len(team_data), 1)
        }
        
        # Normalize features to reasonable ranges
        for key, value in features.items():
            if pd.isna(value) or np.isinf(value):
                features[key] = 0
            elif value < 0:
                features[key] = 0
                
        return features
    
    def generate_realistic_training_data(self, n_samples=3000):
        """Generate more realistic training data"""
        np.random.seed(42)
        
        all_players = self.df['player'].tolist()
        training_data = []
        
        print(f"Generating {n_samples} realistic training samples...")
        
        # Define team templates for more realistic selection
        def create_balanced_team():
            # Select players with different roles
            top_batsmen = self.df.nlargest(50, 'runs')['player'].tolist()
            top_bowlers = self.df.nlargest(50, 'wickets')['player'].tolist()
            all_rounders = self.df[(self.df['runs'] >= 300) & (self.df['wickets'] >= 15)]['player'].tolist()
            
            team = []
            
            # Add 4-5 batsmen
            available_batsmen = [p for p in top_batsmen if p not in team]
            team.extend(np.random.choice(available_batsmen, min(5, len(available_batsmen)), replace=False))
            
            # Add 4-5 bowlers
            available_bowlers = [p for p in top_bowlers if p not in team]
            team.extend(np.random.choice(available_bowlers, min(4, len(available_bowlers)), replace=False))
            
            # Add 1-2 all-rounders
            available_ar = [p for p in all_rounders if p not in team]
            if available_ar:
                team.extend(np.random.choice(available_ar, min(2, len(available_ar)), replace=False))
            
            # Fill remaining spots
            remaining_players = [p for p in all_players if p not in team]
            needed = 11 - len(team)
            if needed > 0 and len(remaining_players) >= needed:
                team.extend(np.random.choice(remaining_players, needed, replace=False))
            
            return team[:11]
        
        for i in range(n_samples):
            if i % 500 == 0:
                print(f"Generated {i} samples...")
                
            try:
                # Create two balanced teams
                team1_players = create_balanced_team()
                team2_players = create_balanced_team()
                
                # Ensure no overlap
                while len(set(team1_players) & set(team2_players)) > 0:
                    team2_players = create_balanced_team()
                
                # Calculate features
                team1_features = self.calculate_team_features(team1_players)
                team2_features = self.calculate_team_features(team2_players)
                
                if team1_features and team2_features:
                    # Create comparative features
                    combined_features = {}
                    
                    for key in team1_features.keys():
                        combined_features[f'team1_{key}'] = team1_features[key]
                        combined_features[f'team2_{key}'] = team2_features[key]
                        combined_features[f'diff_{key}'] = team1_features[key] - team2_features[key]
                        
                        # Safe ratio calculation
                        if team2_features[key] != 0:
                            combined_features[f'ratio_{key}'] = team1_features[key] / team2_features[key]
                        else:
                            combined_features[f'ratio_{key}'] = 1 if team1_features[key] == 0 else 2
                    
                    # Calculate win probability based on multiple factors
                    strength_diff = (
                        combined_features['diff_avg_player_score'] * 0.3 +
                        combined_features['diff_batting_strength'] * 0.2 +
                        combined_features['diff_bowling_strength'] * 0.2 +
                        combined_features['diff_team_experience'] * 0.15 +
                        combined_features['diff_balance_score'] * 0.15
                    )
                    
                    # Add controlled randomness
                    random_factor = np.random.normal(0, 0.1)
                    win_probability = 1 / (1 + np.exp(-(strength_diff + random_factor)))
                    
                    # Ensure some variability
                    win_probability = np.clip(win_probability, 0.1, 0.9)
                    
                    winner = 1 if np.random.random() < win_probability else 0
                    combined_features['winner'] = winner
                    
                    training_data.append(combined_features)
                    
            except Exception as e:
                continue
        
        print(f"Successfully generated {len(training_data)} training samples")
        return pd.DataFrame(training_data)
    
    def create_optimized_model(self, input_dim):
        """Create an optimized neural network"""
        model = Sequential([
            Dense(256, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            Dropout(0.1),
            
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self):
        """Train the neural network model"""
        print("Generating training data...")
        training_df = self.generate_realistic_training_data(n_samples=4000)
        
        if len(training_df) < 100:
            raise ValueError("Insufficient training data generated")
        
        # Prepare features
        feature_cols = [col for col in training_df.columns if col != 'winner']
        self.feature_columns = feature_cols
        
        X = training_df[feature_cols]
        y = training_df['winner']
        
        # Handle any remaining NaN or inf values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        print(f"Training data shape: {X.shape}")
        print(f"Feature columns: {len(self.feature_columns)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create model
        self.model = self.create_optimized_model(X_train_scaled.shape[1])
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
        )
        
        print("Training neural network...")
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=50,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluate
        y_pred_proba = self.model.predict(X_test_scaled)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel Performance:")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Training completed successfully!")
        
        return history
    
    def predict_match_winner(self, team1_players, team2_players):
        """Predict match winner with comprehensive analysis"""
        if self.model is None:
            print("Training model...")
            self.train_model()
        
        # Validate inputs
        if not isinstance(team1_players, list) or not isinstance(team2_players, list):
            return "Error: Team players must be provided as lists"
        
        # Filter valid players
        valid_players = set(self.df['player'].values)
        team1_valid = [p for p in team1_players if p in valid_players]
        team2_valid = [p for p in team2_players if p in valid_players]
        
        invalid_team1 = [p for p in team1_players if p not in valid_players]
        invalid_team2 = [p for p in team2_players if p not in valid_players]
        
        if invalid_team1:
            print(f"Warning: Invalid players in Team 1: {invalid_team1}")
        if invalid_team2:
            print(f"Warning: Invalid players in Team 2: {invalid_team2}")
        
        if len(team1_valid) < 5 or len(team2_valid) < 5:
            return "Error: Each team needs at least 5 valid players"
        
        # Calculate features
        team1_features = self.calculate_team_features(team1_valid)
        team2_features = self.calculate_team_features(team2_valid)
        
        if not team1_features or not team2_features:
            return "Error: Could not calculate team features"
        
        # Create input features
        combined_features = {}
        for key in team1_features.keys():
            combined_features[f'team1_{key}'] = team1_features[key]
            combined_features[f'team2_{key}'] = team2_features[key]
            combined_features[f'diff_{key}'] = team1_features[key] - team2_features[key]
            
            if team2_features[key] != 0:
                combined_features[f'ratio_{key}'] = team1_features[key] / team2_features[key]
            else:
                combined_features[f'ratio_{key}'] = 1 if team1_features[key] == 0 else 2
        
        # Prepare feature vector
        feature_vector = []
        for col in self.feature_columns:
            feature_vector.append(combined_features.get(col, 0))
        
        feature_vector = np.array([feature_vector])
        feature_vector = self.scaler.transform(feature_vector)
        
        # Make prediction
        win_probability = self.model.predict(feature_vector)[0][0]
        
        # Comprehensive result
        result = {
            'winner': 'Team 1' if win_probability > 0.5 else 'Team 2',
            'confidence': max(win_probability, 1 - win_probability) * 100,
            'team1_win_prob': win_probability * 100,
            'team2_win_prob': (1 - win_probability) * 100,
            'team1_players': team1_valid,
            'team2_players': team2_valid,
            'team1_features': team1_features,
            'team2_features': team2_features,
            'analysis': {
                'batting_advantage': 'Team 1' if team1_features['batting_strength'] > team2_features['batting_strength'] else 'Team 2',
                'bowling_advantage': 'Team 1' if team1_features['bowling_strength'] > team2_features['bowling_strength'] else 'Team 2',
                'experience_advantage': 'Team 1' if team1_features['team_experience'] > team2_features['team_experience'] else 'Team 2',
                'balance_advantage': 'Team 1' if team1_features['balance_score'] > team2_features['balance_score'] else 'Team 2'
            }
        }
        
        return result

# Interactive Interface
def interactive_predictor():
    """Interactive interface for the corrected predictor"""
    print("=== Corrected IPL Neural Network Predictor ===\n")
    
    # Initialize predictor with the dataset
    predictor = CorrectedIPLNeuralPredictor(data_path='IPL-Player-Stat.csv')
    
    def search_players(query):
        matches = predictor.df[predictor.df['player'].str.contains(query, case=False, na=False)]['player'].tolist()
        return matches[:10]
    
    def build_team(team_name):
        print(f"\n--- Building {team_name} ---")
        team = []
        
        while len(team) < 11:
            print(f"\nCurrent {team_name} ({len(team)}/11): {team}")
            
            query = input(f"Search for player {len(team)+1} (or 'done' if finished with at least 5): ").strip()
            
            if query.lower() == 'done':
                if len(team) >= 5:
                    break
                else:
                    print("Need at least 5 players. Continue adding...")
                    continue
            
            matches = search_players(query)
            if matches:
                print("Found players:")
                for i, player in enumerate(matches, 1):
                    print(f"{i}. {player}")
                
                try:
                    choice = int(input("Select player number (0 to search again): "))
                    if 1 <= choice <= len(matches):
                        selected_player = matches[choice-1]
                        if selected_player not in team:
                            team.append(selected_player)
                            print(f"✓ Added {selected_player} to {team_name}")
                        else:
                            print("Player already in team!")
                    elif choice == 0:
                        continue
                    else:
                        print("Invalid choice!")
                except ValueError:
                    print("Please enter a valid number!")
            else:
                print("No players found. Try a different search term.")
        
        return team
    
    # Build teams
    team1 = build_team("Team 1")
    team2 = build_team("Team 2")
    
    # Make prediction
    print("\n" + "="*50)
    print("MATCH PREDICTION ANALYSIS")
    print("="*50)
    
    result = predictor.predict_match_winner(team1, team2)
    
    if isinstance(result, dict):
        print(f"\n🏏 TEAM LINEUPS:")
        print(f"Team 1: {result['team1_players']}")
        print(f"Team 2: {result['team2_players']}")
        
        print(f"\n🏆 PREDICTION:")
        print(f"Winner: {result['winner']}")
        print(f"Confidence: {result['confidence']:.1f}%")
        print(f"Team 1 Win Probability: {result['team1_win_prob']:.1f}%")
        print(f"Team 2 Win Probability: {result['team2_win_prob']:.1f}%")
        
        print(f"\n📊 DETAILED ANALYSIS:")
        analysis = result['analysis']
        print(f"Batting Advantage: {analysis['batting_advantage']}")
        print(f"Bowling Advantage: {analysis['bowling_advantage']}")
        print(f"Experience Advantage: {analysis['experience_advantage']}")
        print(f"Team Balance Advantage: {analysis['balance_advantage']}")
        
        print(f"\n📈 KEY METRICS:")
        t1_features = result['team1_features']
        t2_features = result['team2_features']
        print(f"Average Player Score: Team 1 ({t1_features['avg_player_score']:.3f}) vs Team 2 ({t2_features['avg_player_score']:.3f})")
        print(f"Batting Strength: Team 1 ({t1_features['batting_strength']:.2f}) vs Team 2 ({t2_features['batting_strength']:.2f})")
        print(f"Bowling Strength: Team 1 ({t1_features['bowling_strength']:.2f}) vs Team 2 ({t2_features['bowling_strength']:.2f})")
        print(f"All-rounders: Team 1 ({t1_features['num_all_rounders']}) vs Team 2 ({t2_features['num_all_rounders']})")
        
    else:
        print(f"❌ {result}")

# Example usage with predefined teams
def demo_prediction():
    """Demonstrate with example teams"""
    print("=== Demo Prediction ===\n")
    
    predictor = CorrectedIPLNeuralPredictor(data_path='IPL-Player-Stat.csv')
    
    # Example teams with top performers
    rcb_example = [
    'F du Plessis',
    'V Kohli',
    'GJ Maxwell',
    'D Padikkal',
    'Anuj Rawat',
    'RM Patidar',
    'Mahipal Lomror',  # not found exactly
    'SS Prabhudessai',
    'DK Karthik',
    'Suyash Prabhudessai',  # if alternate
    'C de Grandhomme',
    'Wayne Parnell',  # not found
    'Mohammed Siraj',
    'Harshal Patel',  # not found
    'KV Sharma',  # Karn Sharma
    'Akash Deep',
    'Shahbaz Ahmed',
    'Michael Bracewell',  # not found
    'Josh Hazlewood',
    'Reece Topley',  # not found
    'DW Steyn',  # previously
    'YS Chahal',  # previously
    'AB de Villiers',  # previously
]
    
    pbks_example = [
    'S Dhawan',
    'Jitesh Sharma',  # not found
    'Shahrukh Khan',  # as 'M Shahrukh Khan'
    'LS Livingstone',
    'Sikandar Raza',  # not found
    'Liam Livingstone',  # already covered
    'Sam Curran',
    'Kagiso Rabada',  # 'K Rabada'
    'Arshdeep Singh',
    'Harpreet Brar',
    'Rahul Chahar',  # not found
    'Rishi Dhawan',  # possibly 'R Dhawan'
    'Nathan Ellis',  # 'NT Ellis'
    'Shikhar Dhawan',
    'Prabhsimran Singh',  # not found
    'Atharva Taide',  # not found
    'Bhanuka Rajapaksa',  # 'PBB Rajapaksa'
    'Matt Short',  # 'DJM Short'
    'Raj Bawa',  # 'RA Bawa'
    'Baltej Singh',  # not found
    'Karanveer Singh',  # found
    'R Tewatia'  # previously
]

    print("Demo Teams:")
    print(f"Team 1: {team1_example}")
    print(f"Team 2: {team2_example}\n")
    
    result = predictor.predict_match_winner(team1_example, team2_example)
    
    if isinstance(result, dict):
        print(f"🏆 Predicted Winner: {result['winner']}")
        print(f"🎯 Confidence: {result['confidence']:.1f}%")
        print(f"📊 Win Probabilities: Team 1 ({result['team1_win_prob']:.1f}%) vs Team 2 ({result['team2_win_prob']:.1f}%)")
    else:
        print(result)

# Run the demo
if __name__ == "__main__":
    # Run demo first
    demo_prediction()
    
    # Uncomment below to run interactive predictor
    # interactive_predictor()

