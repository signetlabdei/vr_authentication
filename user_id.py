import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from tqdm.notebook import tqdm
from sklearn.ensemble import VotingClassifier
import pickle
from IPython.display import display
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE

SEED = 42
LINUX = True
if (LINUX):
    RESULTS_DIR = os.path.join('./id_results/')
    MODELS_DIR = os.path.join('./id_models/')
else:
    RESULTS_DIR = os.path.join('C:\\Users\\sarab\\Desktop\\id_results', 'authentication', 'user_evaluation_1')
    MODELS_DIR = os.path.join('C:\\Users\\sarab\\Desktop\\id_models', 'authentication', 'user_evaluation_1')

# Training flags
TRAIN = False
TRAIN_VOTING = True

# Heigth normalization flag
NORM_HEIGHT = True
# Traffic inclusion flag
TRAFFIC = True

############## Game Identification Class ##############

class GameIdentification:
    """Main identification system class."""
    
    def __init__(self, id_data):
        self.forklift_dataset = id_data['forklift_simulator']
        self.beat_saber_dataset = id_data['beat_saber']
        
        self.forklift_models = {}
        self.beat_saber_models = {}
        self.forklift_test_set = {}
        self.beat_saber_test_set = {}

        self.models = {}
        
        self.performance_metrics = pd.DataFrame(columns=['User_ID', 'Subset_no', 'Game_Type', 'Model_Type', 'Accuracy', 'Precision',
                                                        'Recall', 'F1'])
        self.confusion_matrices = {}
        
    def preprocess_data(self, data):
        """Preprocess data for training/testing."""
        processed_data = data.copy()

        X = processed_data.drop(columns=['time','ID'])
        y = processed_data['ID']
        
        return X, y 
    
    def train_multiple_models(self, filename="normal"):
        """Train and compare multiple identification models."""
        
        self.models = {
            'ET': ExtraTreesClassifier( bootstrap = False, max_depth = None,
                                                            min_samples_split= 2, n_estimators = 800,
                                                            random_state=SEED),
            'RF': RandomForestClassifier(n_estimators=100, random_state=SEED),
            'LR': LogisticRegression(random_state=SEED, max_iter=1000),
            'XGB': XGBClassifier(random_state=SEED),
            'LGBM': LGBMClassifier(random_state=SEED, verbose=-1)
        }

        # Add the VotingClassifier
        voting_classifiers = [
            ('ET', self.models.get('ET')),  # ExtraTreesClassifier
            ('RF', self.models.get('RF')),  # RandomForestClassifier
            ('LR', self.models.get('LR')),  # LogisticRegression
            ('XGB', self.models.get('XGB')),  # XGBoost 
            ('LGBM', self.models.get('LGBM')),  # LGBMClassifier
            ] 
        
        
        voting_classifier = VotingClassifier(
            estimators=voting_classifiers,
            voting='soft',
            n_jobs=-1
        )
        
        self.models["VotingClassifier"] = voting_classifier
        
        self.train_game_models(
            self.forklift_dataset,
            self.forklift_models, 
            'forklift_simulator', 
            self.models,
            filename=filename
        )
        
        self.train_game_models(
            self.beat_saber_dataset,
            self.beat_saber_models, 
            'beat_saber', 
            self.models,
            filename=filename
        )

    def split_data(self, data, train_minutes=8):

        data['time'] = data['time'].map(int)
        train_data = data[data['time']<train_minutes] 
        test_data = data[data['time']>=train_minutes]

        return train_data, test_data
    
    def train_game_models(self, game_dataset, model_dict, game_type, models, test_intruders_count = 4, apply_smote=False, filename="normal"):
        """Train models for a specific game."""
        models_dir = os.path.join(MODELS_DIR, game_type)
        os.makedirs(models_dir, exist_ok=True)
        
        dataset_metrics = []
        np.random.seed(SEED)

        self.confusion_matrices[game_type] = {}
        train_data, test_data = self.split_data(game_dataset)

        # Sample dataset info for one of the sets
        dataset_info = {
            'Game_Type': game_type,
            'Train_Size': train_data.shape[0],
            'Test_Size': test_data.shape[0]
        }

        dataset_metrics.append(dataset_info)

            
        accuracy = []
        recall = []
        precision = []
        confusion = []
        f1 = []

        X_train, y_train = self.preprocess_data(train_data)
        X_test, y_test = self.preprocess_data(test_data)

        for model_name, model in models.items():
            #print(model_name)

            model.fit(X_train, y_train)

            model_dict[model_name] = model
            model_path = os.path.join(models_dir, f'{model_name}_{filename}.joblib')
            os.makedirs(models_dir, exist_ok=True)

            joblib.dump(model, model_path)

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0, average='macro')
            recall = recall_score(y_test, y_pred, zero_division=0, average='macro')
            f1 = f1_score(y_test, y_pred, zero_division=0, average='macro')

            self.confusion_matrices[game_type][model_name] = confusion_matrix(y_test, y_pred)


            self.performance_metrics = pd.concat([
                self.performance_metrics,
                pd.DataFrame({
                    'Game_Type': [game_type],
                    'Model_Type': [model_name],
                    'Accuracy': [accuracy],
                    'Precision': [precision],
                    'Recall': [recall],
                    'F1': [f1]
                })
            ], ignore_index=True)
                        
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        dataset_metrics_df = pd.DataFrame(dataset_metrics)
        dataset_metrics_path = os.path.join(RESULTS_DIR, f'dataset_metrics_{game_type}_{filename}.csv')
        dataset_metrics_df.to_csv(dataset_metrics_path, index=False) # - problem to solve: path too long

        model_metrics_path = os.path.join(RESULTS_DIR, f'model_metrics_{filename}.csv')
        self.performance_metrics.to_csv(model_metrics_path, index=False)

        model_metrics_path = os.path.join(RESULTS_DIR, f'confusion_{filename}.pkl')
        with open(model_metrics_path, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(self.confusion_matrices, outp, pickle.HIGHEST_PROTOCOL)

    def get_performance_summary(self):
        """Get performance summary."""
        return self.performance_metrics

    def get_confusion(self, model_name):
        """Get performance summary."""
        return self.confusion_matrices[game][model_name]
    
    def identify_user(self, gameplay_data, model=None):
        """Identify a user based on gameplay data."""
        
        res = model.predict_proba(gameplay_data)
        
        return res
    

    def perform_voting(self, voting_threshold=0.5, model_name="ET", filename="initial", output_filename = 'id_voting_results_nofilename.pkl'):
        
        output_prefix = f"voting_effect_{filename}_{model_name}"
        
        results = []
        user_results_by_game = {'forklift_simulator': {}, 'beat_saber': {}}
        
        for game_type in ['beat_saber', 'forklift_simulator']:
            print(f'Game: {game_type}')
            if game_type == 'forklift_simulator':
                game_dataset = self.forklift_dataset
            else:
                game_dataset = self.beat_saber_dataset

            train_data, test_data = self.split_data(game_dataset)

            model_path = os.path.join(MODELS_DIR, game_type, f'{model_name}_{filename}.joblib')
            loaded_model = joblib.load(model_path)
            y_test = test_data['ID'].values

            user_results_by_game[game_type] = []

            samples = len(np.where(y_test==0)[0])

            for n_samples in range(1, int(np.floor(samples / 3))):
                acc = 0
                tot = 0
                print(n_samples)
                for user in range(30):
                    user_data = test_data[test_data['ID'] == user]
                    user_copy = user_data.copy()
                    user_features = user_copy.drop(columns=['ID', 'time'])
                    preds = self.identify_user(user_features, model = loaded_model)
                    for offset in range(0, samples + 1 - n_samples, n_samples): # 24 so that when n sample is 1 range is between 0 and 24 (i.e., 0-23), and when n samples is 23 range is between 0 and 2
                        group = preds[offset: offset + n_samples]
                        id_dist = np.zeros(30)
                        for i in range(n_samples):
                            id_dist += group[i]

                        id_dist /= np.sum(id_dist)

                        if (np.argmax(id_dist) == user):
                            acc += 1
                        tot +=1

                accuracy = acc / tot


                result = {
                    'Game_Type': game_type,
                    'Model_Type': model_name,
                    'Window_size': n_samples,
                    'Accuracy': accuracy,
                }

                user_results_by_game[game_type].append(result)
                results.append(result)
        
        results_df = pd.DataFrame(results)
        
        csv_path = os.path.join(RESULTS_DIR, f'{output_prefix}.csv')
        results_df.to_csv(csv_path, index=False)
        
        with open(output_filename, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(results_df, outp, pickle.HIGHEST_PROTOCOL)

        return results_df
    
    def visualize_voting_results(self,results):
                                    
        accuracy_mean = results.groupby(['Game_Type', 'Window_size','Model_Type']).mean().reset_index()
        for game_type in ['forklift_simulator', 'beat_saber']:

            test_game = accuracy_mean.loc[accuracy_mean['Game_Type']==game_type]
            plt.figure(figsize=(14, 8))

            plt.plot(test_game['Window_size'], test_game['Accuracy'], label=game_type)

            plt.title(f'Overall Accuracy by Voting Samples - {game_type.replace("_", " ").title()}', fontsize=14)
            plt.xlabel('Number of Samples Used for Voting', fontsize=12)
            plt.ylabel('Overall Accuracy', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='lower right', bbox_to_anchor=(1, 0), ncol=3)
            plt.ylim(0.4, 1.01)

            plt.tight_layout()
            plt.show()

    
############## Functions ##############
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def prepare_id_data(forklift_simulator, beat_saber, include_traffic=False):
    """Prepare data for authentication systems."""
    
    data_forklift_simulator = forklift_simulator.copy()
    data_forklift_simulator.drop(columns=data_forklift_simulator.columns[0], axis=1,  inplace=True) # first column contains row index
    data_beat_saber = beat_saber.copy()
    data_beat_saber.drop(columns=data_beat_saber.columns[0], axis=1,  inplace=True) # first column contains row index
    
    if not include_traffic: 

        cols_to_drop = []
        cols = forklift_simulator.columns
        for col in cols:
            if 'Traffic' in col:
                cols_to_drop.append(col)

        data_forklift_simulator = data_forklift_simulator.drop(columns=cols_to_drop)
        data_beat_saber = data_beat_saber.drop(columns=cols_to_drop)

    data_forklift_simulator['ID'] -= 30

    return {
        'forklift_simulator': data_forklift_simulator,
        'beat_saber': data_beat_saber
        }

############## Main code ##############

def main():
    # Load data
    if (LINUX):
        forklift_simulator = pd.read_csv('./data/processed/slow_features.csv').fillna(0)
        beat_saber = pd.read_csv('./data/processed/fast_features.csv').fillna(0)
    else:
        forklift_simulator = pd.read_csv('G:\\Il mio Drive\\Esperimenti Oculus\\VR authentication\\vr-user-identification-authentication-main\\vr-user-identification-authentication-main\\data\\processed\\users\\slow_features.csv').fillna(0)
        beat_saber = pd.read_csv('G:\\Il mio Drive\\Esperimenti Oculus\\VR authentication\\vr-user-identification-authentication-main\\vr-user-identification-authentication-main\\data\\processed\\users\\fast_features.csv').fillna(0)


    forklift_simulator = forklift_simulator[forklift_simulator["ID"].isin(range(30,60))]
    beat_saber = beat_saber[beat_saber["ID"].isin(range(0,30))]
    #traffic_forklift_simulator =traffic_forklift_simulator[traffic_forklift_simulator["ID"].isin(range(30,60))]
    #traffic_beat_saber = traffic_beat_saber[traffic_beat_saber["ID"].isin(range(0,30))]


    if TRAFFIC:
        filename = 'MOV_TRAFFIC'
    else:
        filename = 'MOV'

    if NORM_HEIGHT:
        filename += '_NO_HEIGHT'
        output_filename = 'id_mov_traffic_without_height.pkl'
        voting_results = 'id_voting_results_mov_traffic_without_height_confidence.pkl'
        for col in forklift_simulator.columns.values:
            if "PosY" in col and "Acc" not in col and "Vel" not in col:
                for id in range(30):
                    beat_saber.loc[beat_saber['ID'] == id,col] = beat_saber.loc[beat_saber['ID'] == id,col] / np.mean(beat_saber.loc[beat_saber['ID'] == id,col])
                for id in range(30,60):
                    forklift_simulator.loc[forklift_simulator['ID'] == id,col] = forklift_simulator.loc[forklift_simulator['ID'] == id,col] / np.mean(forklift_simulator.loc[forklift_simulator['ID'] == id,col])
    else:
        output_filename = 'id_mov_traffic.pkl'
        voting_results = 'id_voting_results_mov_traffic_confidence.pkl'

    

    # Prepare data for identification
    identification_data = prepare_id_data(
        forklift_simulator,
        beat_saber,
        include_traffic=TRAFFIC,
    )

    if TRAIN:
        # Create game identifier instance
        identifier = GameIdentification(identification_data)
        # Train multiple models
        identifier.train_multiple_models(filename = filename)

        # Save authenticator
        save_object(identifier, output_filename)
    else:
        with open(output_filename, 'rb') as file:
            identifier = pickle.load(file)


    # Get performance metrics
    performance_metrics = identifier.get_performance_summary()

    # Show average performance by model type
    model_performance = performance_metrics.groupby(['Game_Type', 'Model_Type']).mean().reset_index()
    print("Average model performance across all users:")
    display(model_performance[['Game_Type', 'Model_Type', 'Accuracy', 'Precision', 'Recall', 'F1']])

    # Implement voting
    if TRAIN_VOTING:
        results = identifier.perform_voting(voting_threshold=0.5, model_name="ET", filename=filename, output_filename=voting_results)
    else:
        with open(voting_results, 'rb') as file:
            results = pickle.load(file)

    identifier.visualize_voting_results(results)

if __name__ == '__main__':
    main()
