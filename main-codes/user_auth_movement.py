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
import shap

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE

SEED = 42
RESULTS_DIR = os.path.join('C:\\Users\\root\\Desktop\\results', 'authentication', 'user_evaluation')
MODELS_DIR = os.path.join('G:\\Il mio Drive\\Esperimenti Oculus\\VR authentication\\vr-user-identification-authentication-main\\vr-user-identification-authentication-main\\models', 'authentication', 'user_evaluation')

# Training flags
TRAIN = True
COMPUTE_VOTING = False 
PERFORM_VOTING = False

# Heigth normalization flag
NORM_HEIGHT = True

# Shap flag
SHAP = False # perform shap - note: if this is true but all other shap flags are false, pre-computed all users shap values are used
SHAP_ALL = False # Perform shap for all users, computing the shap values
SHAP_USER_BEAT = False # Perform shap for specific user, computing the shap values
SHAP_USER_FORKLIFT = False # Perform shap for specific user, computing the shap values
USER = 22 # user ID from 0 to 59 - note: beat saber users go from 0 to 29, forklift users go from 30 to 59

############## Game Authenticator Class ##############

class GameAuthenticator:
    """Main authentication system class."""
    
    def __init__(self, authentication_data):
        self.forklift_datasets = authentication_data['forklift_simulator']['datasets']
        self.forklift_user_ids = authentication_data['forklift_simulator']['user_ids']
        
        self.beat_saber_datasets = authentication_data['beat_saber']['datasets']
        self.beat_saber_user_ids = authentication_data['beat_saber']['user_ids']
        
        self.forklift_models = {}
        self.beat_saber_models = {}
        self.forklift_test_sets = {}
        self.beat_saber_test_sets = {}

        self.models = {}
        
        self.performance_metrics = pd.DataFrame(columns=['User_ID', 'Subset_no', 'Test_set', 'Game_Type', 'Model_Type', 'Accuracy', 'Precision', 
                                                        'Recall', 'F1', 'AUC', 'FAR', 'FRR',  'tn', 'fn', 'fp', 'tp'])
        
    def preprocess_data(self, data):
        """Preprocess data for training/testing."""
        processed_data = data.copy()

        X = processed_data.drop(columns=['time','is_genuine', 'original_id'])
        y = processed_data['is_genuine']
        
        return X, y 
    
    def train_multiple_models(self, filename="normal", apply_smote=False):
        """Train and compare multiple authentication models."""
        
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
            self.forklift_datasets, 
            self.forklift_user_ids, 
            self.forklift_models, 
            'forklift_simulator', 
            self.models,
            apply_smote=apply_smote,
            filename=filename
        )
        
        self.train_game_models(
            self.beat_saber_datasets, 
            self.beat_saber_user_ids, 
            self.beat_saber_models, 
            'beat_saber', 
            self.models,
            apply_smote=apply_smote,
            filename=filename
        )

    def split_data(self, data, test_only_intruder_ids, train_intruder_ids, train_minutes=8, train_sample = 250, test_train_sample = 24, test_sample = 6):
        # train_sample: number of samples for the intruders in the training set
        # test_train_sample: number of samples for the intruders in the test set when there are trained intruders (overall)
        # test_sample: number of samples for the intruders in the test set when intruders are not in the training set (per intruder)

        data['time'] = data['time'].map(int)
                
        train_data = data[data['time']<train_minutes] 
        test_data = data[data['time']>=train_minutes]

        train_genuine = train_data[train_data['is_genuine'] == 1] 
        test_genuine = test_data[test_data['is_genuine'] ==1 ] 
        
        np.random.seed(SEED)
        train_ids = []
        train_impostors_sampled = []

        for train_intruder_id in train_intruder_ids:
            train_impostor = train_data[train_data['original_id'].isin(train_intruder_id)]
            id = np.random.choice(range(len(train_impostor)),train_sample,replace=False) 
            train_ids.append(id)
            train_impostors_sampled.append(train_impostor.iloc[id])

        test_only_impostors = []
        test_only_impostors_sampled = []
        test_only_window_ids = []
        for test_only_intruder_id in test_only_intruder_ids:
            test_only_impostors.append(test_data[test_data['original_id'].isin(test_only_intruder_id)]) 
            test_only_set = pd.DataFrame() 
            test_only_window_id = []
            for single_intruder_id in test_only_intruder_id:
                test_only_single_impostor = test_data[test_data['original_id'] == single_intruder_id]
                id = np.random.choice(range(len(test_only_single_impostor)),test_sample,replace=False) 
                test_only_window_id.extend(list(id))
                test_only_set = pd.concat([test_only_set, test_only_single_impostor.iloc[id]], ignore_index=True)
            test_only_impostors_sampled.append(test_only_set)
            test_only_window_ids.append(np.array(test_only_window_id))
        
        test_trained_impostors = []
        test_trained_impostors_sampled = []
        test_trained_window_ids = []
        for test_trained_intruder_id in train_intruder_ids:
            test_trained_impostor = test_data[test_data['original_id'].isin(test_trained_intruder_id)] 
            id = np.random.choice(range(len(test_trained_impostor)),test_train_sample,replace=False) 
            test_trained_impostors_sampled.append(test_trained_impostor.iloc[id])
            window_ids = np.mod(id,24) # there are 24 samples per user inside test_trained_impostor
            test_trained_window_ids.append(window_ids)
            test_trained_impostors.append(test_trained_impostor)


        train_data_complete = [pd.concat([train_genuine, train_impostor_sampled]) for train_impostor_sampled in train_impostors_sampled]
        test_data_only = [pd.concat([test_genuine, test_only_impostor_sampled]) for test_only_impostor_sampled in test_only_impostors_sampled]
        test_data_trained = [pd.concat([test_genuine, test_trained_impostor_sampled]) for test_trained_impostor_sampled in test_trained_impostors_sampled]
        
        # De-comment for obtaining the clean files needed for spoofing

        # with open(RESULTS_DIR+'\\spoofing_data\\test_data_only_pre_attack_'+ game_type + str(user_id) + '.pkl', 'wb') as outp:  # Overwrites any existing file.
        #     pickle.dump(test_data_only, outp, pickle.HIGHEST_PROTOCOL)
        # with open(RESULTS_DIR+'\\spoofing_data\\test_data_trained_pre_attack_' + game_type + str(user_id) + '.pkl', 'wb') as outp:  # Overwrites any existing file.
        #     pickle.dump(test_data_trained, outp, pickle.HIGHEST_PROTOCOL)

        # with open(RESULTS_DIR+'\\spoofing_data\\test_only_window_ids_pre_attack_' + game_type + str(user_id) + '.pkl', 'wb') as outp:  # Overwrites any existing file.
        #     pickle.dump(test_only_window_ids, outp, pickle.HIGHEST_PROTOCOL)

        # with open(RESULTS_DIR+'\\spoofing_data\\test_trained_window_ids_pre_attack_' + game_type + str(user_id) + '.pkl', 'wb') as outp:  # Overwrites any existing file.
        #     pickle.dump(test_trained_window_ids, outp, pickle.HIGHEST_PROTOCOL)

        return train_data_complete, test_data_only, test_data_trained
    
    def train_game_models(self, game_datasets, user_ids, model_dict, game_type, models, test_intruders_count = 4, apply_smote=False, filename="normal"):
        """Train models for a specific game."""
        models_dir = os.path.join(MODELS_DIR, game_type)
        
        os.makedirs(models_dir, exist_ok=True)
        
        dataset_metrics = []
        np.random.seed(SEED)

        for user_id in tqdm(user_ids):
            model_dict[user_id] = {}
            
            
            user_models_dir = os.path.join(models_dir, f'user_{user_id}')
            os.makedirs(user_models_dir, exist_ok=True)
            
            data = game_datasets[user_id]
            
            all_intruder_ids = np.array([id for id in user_ids if id != user_id])
            
            
            np.random.shuffle(all_intruder_ids) 
            groups = [all_intruder_ids[i:i+test_intruders_count] for i in range(0, len(all_intruder_ids), test_intruders_count)][:-1] 
            
            test_only_intruders = np.array([groups[i] for i in range(len(groups))])
            train_intruders = []
            for i in range(len(groups)):
                train_intruders.append(np.array([id for id in all_intruder_ids if id not in test_only_intruders[i]]))
    
            train_intruders = np.array(train_intruders)

            train_data_all, test_only_data_all, test_trained_data_all = self.split_data(data,test_only_intruders,train_intruders)

            # Sample dataset info for one of the sets
            dataset_info = {
                'User_ID': user_id,
                'Game_Type': game_type,
                'Train_Size': len(train_data_all[0]),
                'Train_Genuine': sum(train_data_all[0]['is_genuine'] == 1),
                'Train_Impostor': sum(train_data_all[0]['is_genuine'] == 0),
                'Test_Size': len(test_only_data_all[0]),
                'Test_Genuine': sum(test_only_data_all[0]['is_genuine']==1),
                'Test_Impostor_New_IDs': sum(test_only_data_all[0]['is_genuine']==0),
                'Test_Impostor_Train_IDs': sum(test_trained_data_all[0]['is_genuine']==0)
            }
            

            for t in range(2): #iterate over the two test sets

                if game_type == 'forklift_simulator' and user_id == user_ids[0]:
                    self.forklift_test_sets[f'test_{t}']={}
                elif game_type == 'beat_saber' and user_id == user_ids[0]:
                    self.beat_saber_test_sets[f'test_{t}']={}

                if t == 0:
                    test_data_all = test_only_data_all
                else:
                    test_data_all = test_trained_data_all

                accuracy = []
                recall = []
                precision = []
                far = []
                frr = []
                f1 = []
                for i in range(len(train_data_all)): # iterate over the data subsets

                    if game_type == 'forklift_simulator' and user_id == user_ids[0]:
                        self.forklift_test_sets[f'test_{t}'][f'subset_{i}']={}
                    elif 'beat_saber' and user_id == user_ids[0]:
                        self.beat_saber_test_sets[f'test_{t}'][f'subset_{i}']={}

                    test_data = test_data_all[i]
                    train_data = train_data_all[i]

                    if game_type == 'forklift_simulator':
                        self.forklift_test_sets[f'test_{t}'][f'subset_{i}'][user_id] = test_data
                    else:
                        self.beat_saber_test_sets[f'test_{t}'][f'subset_{i}'][user_id] = test_data
                                    
                    X_train, y_train = self.preprocess_data(train_data)
                    X_test, y_test = self.preprocess_data(test_data)
                    
                    genuine_count_before = sum(y_train == 1)
                    impostor_count_before = sum(y_train == 0)
                    dataset_info['Train_Genuine_Before_SMOTE'] = genuine_count_before
                    dataset_info['Train_Impostor_Before_SMOTE'] = impostor_count_before
                    
                    if apply_smote:
                        
                        smote = SMOTE(random_state=SEED)
                        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                        X_train = pd.DataFrame(X_train_resampled, columns=X_train.columns)
                        y_train = pd.Series(y_train_resampled)
                        
                        genuine_count_after = sum(y_train == 1)
                        impostor_count_after = sum(y_train == 0)
                        dataset_info['Train_Genuine_After_SMOTE'] = genuine_count_after
                        dataset_info['Train_Impostor_After_SMOTE'] = impostor_count_after
                        dataset_info['SMOTE_Applied'] = True
                            
                    else:
                        dataset_info['Train_Genuine_After_SMOTE'] = genuine_count_before
                        dataset_info['Train_Impostor_After_SMOTE'] = impostor_count_before
                        dataset_info['SMOTE_Applied'] = False
                        
                    dataset_metrics.append(dataset_info)
                    
                    for model_name, model in models.items():
                                                    
                        model.fit(X_train, y_train)
                        
                        model_dict[user_id][model_name] = model
                        model_path = os.path.join(user_models_dir, f'{model_name}_{filename}_test_{t}_subset_{i}.joblib')
                        os.makedirs(user_models_dir, exist_ok=True)
                        
                        joblib.dump(model, model_path)
                                                
                        y_pred = model.predict(X_test)
                        
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, zero_division=0)
                        recall = recall_score(y_test, y_pred, zero_division=0)
                        f1 = f1_score(y_test, y_pred, zero_division=0)
                        
                        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
                        far = fp / (fp + tn) if (fp + tn) > 0 else 0  
                        frr = fn / (fn + tp) if (fn + tp) > 0 else 0  
                        
                        
                        self.performance_metrics = pd.concat([
                            self.performance_metrics,
                            pd.DataFrame({
                                'User_ID': [user_id],
                                'Subset_no': [i],
                                'Test_set':[t],
                                'Game_Type': [game_type],
                                'Model_Type': [model_name],
                                'Accuracy': [accuracy],
                                'Precision': [precision],
                                'Recall': [recall],
                                'F1': [f1],
                                'FAR': [far],
                                'FRR': [frr],
                                'tn': tn,
                                'fn': fn,
                                'fp': fp,
                                'tp': tp
                            })
                        ], ignore_index=True)
                        
                    
        
        
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        dataset_metrics_df = pd.DataFrame(dataset_metrics)
        dataset_metrics_path = os.path.join(RESULTS_DIR, f'dataset_metrics_{game_type}_{filename}.csv')
        dataset_metrics_df.to_csv(dataset_metrics_path, index=False) 

        model_metrics_path = os.path.join(RESULTS_DIR, f'model_metrics_{filename}.csv')
        self.performance_metrics.to_csv(model_metrics_path, index=False)

    def get_performance_summary(self):
        """Get performance summary."""
        return self.performance_metrics
    
    def authenticate_user(self, gameplay_data, model=None):
        """Authenticate a user based on gameplay data."""

        res = model.predict_proba(gameplay_data) 

        is_genuine = [bool(np.argmax(sample)) for sample in res]
        confidence = res[:,1] # Class 0 = not genuine, class 1 = genuine
        
        return is_genuine, confidence   
    

    def perform_voting(self, voting_threshold=0.5, model_name="ET", filename="initial", output_filename = 'voting_results_nofilename.pkl'):
        
        
        output_prefix = f"voting_effect_{filename}_{model_name}"
        
        results = []
        user_results_by_game = {'forklift_simulator': {}, 'beat_saber': {}}
        
        for game_type in ['beat_saber', 'forklift_simulator']:
            print(f'Game: {game_type}')
            if game_type == 'forklift_simulator':
                game_user_ids = self.forklift_user_ids
                game_test_sets_all = self.forklift_test_sets
            else:
                game_user_ids = self.beat_saber_user_ids
                game_test_sets_all = self.beat_saber_test_sets
            
            for t in range(2):
                print(f'Test: {t}')
                game_test_sets_t = game_test_sets_all[f'test_{t}']
                for i in range(len(game_test_sets_t)):
                    print(f'Subset: {i}')
                    game_test_sets = game_test_sets_t[f'subset_{i}']
                    for user_id in game_user_ids:
                        if user_id not in game_test_sets:
                            continue

                        model_path = os.path.join(MODELS_DIR, game_type, f'user_{user_id}', f'{model_name}_{filename}_test_{t}_subset_{i}.joblib')
                        
                        test_data = game_test_sets[user_id]
                        
                        loaded_model = joblib.load(model_path)
                                                
                        genuine_test = test_data[test_data['is_genuine'] == 1].reset_index(drop=True)
                        
                        impostor_test = test_data[test_data['is_genuine'] == 0].reset_index(drop=True)
                        
                        
                        if game_type == 'forklift_simulator':
                            if user_id not in self.forklift_models:
                                self.forklift_models[user_id] = {}
                            self.forklift_models[user_id][model_name] = loaded_model
                        else:
                            if user_id not in self.beat_saber_models:
                                self.beat_saber_models[user_id] = {}
                            self.beat_saber_models[user_id][model_name] = loaded_model
                        
                        
                        user_results_by_game[game_type][user_id] = []
                        
                        actual_sample_sizes = range(1, int(np.floor(len(genuine_test) / 3))) # so that we have 3 windows to average

                        overall_acc = {}
                        overall_fp = {}
                        overall_fn = {}

                        for n_samples in actual_sample_sizes:
                            
                            
                            acc_per_window = []
                            fp_per_window = []
                            fn_per_window = []

                            
                            genuine_features = genuine_test.drop(columns=['is_genuine', 'original_id', 'time'])
                            _, genuine_p_genuine = self.authenticate_user(genuine_features, model = loaded_model)
                            
                            impostor_features = impostor_test.drop(columns=['is_genuine', 'original_id', 'time'])
                            _, genuine_p_impostor = self.authenticate_user(impostor_features, model = loaded_model)
                            
                            for offset in range(0, len(genuine_test) + 1 - n_samples, n_samples): # 24 so that when n sample is 1 range is between 0 and 24 (i.e., 0-23), and when n samples is 23 range is between 0 and 2 (i.e., 0-1)
                                tp = 0
                                fn = 0
                                fp = 0
                                tn = 0

                                # We need to keep the genuine and impostor data separated as we are voting on increasing windows and the label of data has to stay constant
                                
                                group_genuine = genuine_p_genuine[offset:offset+n_samples]
                                
                                genuine_probs = np.sum(group_genuine)

                                genuine_percentage = genuine_probs / (len(group_genuine))

                                group_impostor = genuine_p_impostor[offset:offset+n_samples]
                               
                                impostor_probs = np.sum(group_impostor)
                                
                                impostor_percentage = impostor_probs / (len(group_impostor)) 
                                if genuine_percentage >= voting_threshold:
                                    tp += 1
                                else:
                                    fn += 1
                                if impostor_percentage >= voting_threshold: # impostor percentage is the probability that the impostor sample is seen as genuine
                                    fp += 1
                                else:
                                    tn += 1

                                acc_per_window.append((tp+tn)/(tp+tn+fp+fn))
                                fp_per_window.append(fp)
                                fn_per_window.append(fn)
                        
                            
                            overall_acc=np.mean(acc_per_window) # average over the windows of the same size
                            overall_fp=np.mean(fp_per_window)
                            overall_fn=np.mean(fn_per_window)

                            result = {
                                'User_ID': user_id,
                                'Subset_no': i,
                                'Test_set':t,
                                'Game_Type': game_type,
                                'Model_Type': model_name,
                                'Window_size': n_samples,
                                'Accuracy': overall_acc,
                                'False_positives': overall_fp,
                                'False_negatives': overall_fn,
                            }
                        
                            user_results_by_game[game_type][user_id].append(result)
                            results.append(result)
        
        results_df = pd.DataFrame(results)
        
        csv_path = os.path.join(RESULTS_DIR, f'{output_prefix}.csv')
        results_df.to_csv(csv_path, index=False)
        
        with open(output_filename, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(results_df, outp, pickle.HIGHEST_PROTOCOL)

        return results_df
    
    def visualize_voting_results(self,results):
                                    
        accuracy_mean = results.groupby(['User_ID','Test_set', 'Game_Type', 'Window_size','Model_Type']).mean().reset_index()
        for t in range(2):
            test = accuracy_mean.loc[accuracy_mean['Test_set']==t]

            for game_type in ['forklift_simulator', 'beat_saber']:

                test_game = test.loc[test['Game_Type']==game_type]
                plt.figure(figsize=(14, 8))
                

                for id in test_game['User_ID'].unique():

                    
                    user_df = test_game.loc[test_game['User_ID']==id]
                    plt.plot(user_df['Window_size'], user_df['Accuracy'], label=f'User {int(id)}')

                    plt.title(f'Overall Accuracy by Voting Samples - {game_type.replace("_", " ").title()} - Test:{t}', fontsize=14)
                    plt.xlabel('Number of Samples Used for Voting', fontsize=12)
                    plt.ylabel('Overall Accuracy', fontsize=12)
                    plt.grid(True, alpha=0.3)
                    plt.legend(loc='lower right', bbox_to_anchor=(1, 0), ncol=3)
                    plt.ylim(0.4, 1.01)

                plt.tight_layout()
                plt.show()

        accuracy_mean_users = accuracy_mean.groupby(['Test_set', 'Game_Type', 'Window_size','Model_Type']).mean().reset_index()
        for t in range(2):
            test_mean = accuracy_mean_users.loc[accuracy_mean_users['Test_set']==t]
            plt.figure(figsize=(14, 8))
            for game_type in ['forklift_simulator', 'beat_saber']:

                test_game_mean = test_mean.loc[test_mean['Game_Type']==game_type]
                plt.plot(test_game_mean['Window_size'], test_game_mean['Accuracy'], label=f'Game: {game_type}')
                plt.title(f'Overall Accuracy by Voting Samples - {game_type.replace("_", " ").title()} - Test:{t}', fontsize=14)
                plt.xlabel('Number of Samples Used for Voting', fontsize=12)
                plt.ylabel('Overall Accuracy', fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.legend(loc='lower right', bbox_to_anchor=(1, 0), ncol=3)
                plt.ylim(0.85, 1.01)

            plt.tight_layout()
            plt.show()

    def perform_shap(self, model_name = 'ET', filename='initial'):
        
        game_values = {}
        for game_type in ['beat_saber', 'forklift_simulator']:
            game_values[game_type] = {}
            print(f'Game: {game_type}')
            if game_type == 'forklift_simulator':
                game_user_ids = self.forklift_user_ids
                game_test_sets_all = self.forklift_test_sets
            else:
                game_user_ids = self.beat_saber_user_ids
                game_test_sets_all = self.beat_saber_test_sets
            
            
            for t in range(2):
                
                print(f'Test: {t}')
                
                game_test_sets_t = game_test_sets_all[f'test_{t}']
                subset_values = []
                for i in range(len(game_test_sets_t)):
                    print(f'Subset: {i}')
                    game_test_sets = game_test_sets_t[f'subset_{i}']
                    for user_id in game_user_ids:
                        if user_id not in game_test_sets:
                            continue

                        model_path = os.path.join(MODELS_DIR, game_type, f'user_{user_id}', f'{model_name}_{filename}_test_{t}_subset_{i}.joblib')
                        
                        test = game_test_sets[user_id]
                        test_data = test.drop(columns=['is_genuine', 'original_id', 'time'])
                        
                        loaded_model = joblib.load(model_path)

                        explainer = shap.TreeExplainer(loaded_model)
                        shap_values = explainer.shap_values(test_data) # is a list of two elements (opposites, one per class), every element is n_sample(48)xn_features(501)
                        
                        subset_values.append(np.mean(np.abs(shap_values[0]), axis=0)) # mean along the samples, note: this iterates over all the users in a subset
                        
                game_values[game_type][t] = np.mean(np.array(subset_values), axis = 0) # mean along the subsets and the users
            
        columns = test_data.columns

        return game_values, columns
    
    def perform_shap_user(self, model_name = 'ET', filename='initial', user = 0, game = 'beat_saber'):
        
        game_values = {}
        game_type = game
        game_values[game_type] = {}
        print(f'Game: {game_type}')
        if game_type == 'forklift_simulator':
            
            game_test_sets_all = self.forklift_test_sets
        else:
            
            game_test_sets_all = self.beat_saber_test_sets
        
        
        for t in range(2):
            
            print(f'Test: {t}')
            
            game_test_sets_t = game_test_sets_all[f'test_{t}']
            subset_values = []
            for i in range(len(game_test_sets_t)):
                print(f'Subset: {i}')
                game_test_sets = game_test_sets_t[f'subset_{i}']

                user_id = user
                model_path = os.path.join(MODELS_DIR, game_type, f'user_{user_id}', f'{model_name}_{filename}_test_{t}_subset_{i}.joblib')
                
                test = game_test_sets[user_id]
                test_data = test.drop(columns=['is_genuine', 'original_id', 'time'])
                
                loaded_model = joblib.load(model_path)

                explainer = shap.TreeExplainer(loaded_model)
                shap_values = explainer.shap_values(test_data) # is a list of two elements (opposites, one per class), every element is n_sample(48)xn_features(501)
                
                subset_values.append(np.mean(np.abs(shap_values[0]), axis=0)) # mean along the samples
                    
            game_values[game_type][t] = np.mean(np.array(subset_values), axis = 0) # mean along the subsets
            
        columns = test_data.columns

        return game_values, columns

############## Functions ##############
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def prepare_auth_data(forklift_simulator, beat_saber, include_traffic=False):
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

        
    forklift_user_ids = np.array(sorted(data_forklift_simulator['ID'].unique()))
    beat_saber_user_ids = np.array(sorted(data_beat_saber['ID'].unique()))
    
    forklift_datasets = {}
    beat_saber_datasets = {}
    
    for user_id in forklift_user_ids:
        user_data = data_forklift_simulator[data_forklift_simulator['ID'] == user_id].drop(columns=['ID'])
        
        genuine_samples = user_data.copy()
        genuine_samples['is_genuine'] = 1
        genuine_samples['original_id'] = user_id
        
        
        impostor_data = data_forklift_simulator[data_forklift_simulator['ID'] != user_id].drop(columns=['ID'])
        impostor_data['is_genuine'] = 0
        impostor_data['original_id']=data_forklift_simulator[data_forklift_simulator['ID'] != user_id]['ID']
        
        
        user_data_combined = pd.concat([genuine_samples, impostor_data], ignore_index=True)
        
        forklift_datasets[user_id] = user_data_combined
    
    for user_id in beat_saber_user_ids:
        user_data = data_beat_saber[data_beat_saber['ID'] == user_id].drop(columns=['ID'])
        
        genuine_samples = user_data.copy()
        genuine_samples['is_genuine'] = 1
        genuine_samples['original_id'] = user_id
        
        impostor_data = data_beat_saber[data_beat_saber['ID'] != user_id].drop(columns=['ID'])
        impostor_data['is_genuine'] = 0
        impostor_data['original_id']=data_beat_saber[data_beat_saber['ID'] != user_id]['ID']
        
        user_data_combined = pd.concat([genuine_samples, impostor_data], ignore_index=True)
        
        beat_saber_datasets[user_id] = user_data_combined
    
    return {
        'forklift_simulator': {
            'datasets': forklift_datasets,
            'user_ids': forklift_user_ids
        },
        'beat_saber': {
            'datasets': beat_saber_datasets,
            'user_ids': beat_saber_user_ids
        }
    }



############## Main code ##############

def main():
    # Load data
    forklift_simulator = pd.read_csv('..\\data\\processed\\slow_features.csv').fillna(0)
    beat_saber = pd.read_csv('..\\data\\processed\\fast_features.csv').fillna(0)


    forklift_simulator = forklift_simulator[forklift_simulator["ID"].isin(range(30,60))]
    beat_saber = beat_saber[beat_saber["ID"].isin(range(0,30))]


    if NORM_HEIGHT:
        filename = 'MOV_NO_HEIGHT'
        shap_filename = 'shap_without_heigth_no_traffic.pkl'
        output_filename = 'auth_mov_without_height.pkl'
        voting_results = 'voting_results_mov_without_height_confidence_complete.pkl'

        #norm_factor_beat = {'id': [], 'col':[], 'val': []} # de-comment if normalization factors are needed for spoofing
        #norm_factor_forklift = {'id': [], 'col':[], 'val': []} # de-comment if normalization factors are needed for spoofing

        for col in forklift_simulator.columns.values:
            if "PosY" in col and "Acc" not in col and "Vel" not in col:
                for id in range(30):
                    mean_value = np.mean(beat_saber.loc[beat_saber['ID'] == id,col])
                    beat_saber.loc[beat_saber['ID'] == id,col] = beat_saber.loc[beat_saber['ID'] == id,col] / mean_value
                    # de-comment if normalization factors are needed for spoofing
                    # norm_factor_beat['val'].append(mean_value)
                    # norm_factor_beat['id'].append(id)
                    # norm_factor_beat['col'].append(col)
                    
                for id in range(30,60):
                    mean_value = np.mean(forklift_simulator.loc[forklift_simulator['ID'] == id,col])
                    forklift_simulator.loc[forklift_simulator['ID'] == id,col] = forklift_simulator.loc[forklift_simulator['ID'] == id,col] / mean_value
                    # de-comment if normalization factors are needed for spoofing
                    #norm_factor_forklift['val'].append(mean_value)
                    #norm_factor_forklift['id'].append(id)
                    #norm_factor_forklift['col'].append(col)
                    
        # de-comment if normalization factors are needed for spoofing
        # norm_factor_beat_pd=pd.DataFrame(norm_factor_beat)
        # norm_factor_forklift_pd=pd.DataFrame(norm_factor_forklift)
        # norm_factor_beat_pd.to_csv('norm_factor_beat.csv', index=False) 
        # norm_factor_forklift_pd.to_csv('norm_factor_forklift.csv', index=False) 

    else:
        filename = 'MOV'
        shap_filename = 'shap_no_traffic.pkl'
        output_filename = 'auth_mov.pkl'
        voting_results = 'voting_results_mov_confidence_complete.pkl'

    

    # Prepare data for authentication
    authentication_data = prepare_auth_data(
        forklift_simulator, 
        beat_saber, 
        include_traffic=False, 
    )

    # Look at the structure of sample users' datasets
    print('---- Dataset structure ----')
    print(f"\nSample Forklift Simulator User Data:")
    sample_forklift_user = authentication_data['forklift_simulator']['user_ids'][0]
    print(f"User ID: {sample_forklift_user}")
    print(f"Data shape: {authentication_data['forklift_simulator']['datasets'][sample_forklift_user].shape}")
    print(f"Class balance:")
    print(authentication_data['forklift_simulator']['datasets'][sample_forklift_user]['is_genuine'].value_counts())

    print(f"\nSample Beat Saber User Data:")
    sample_beat_saber_user = authentication_data['beat_saber']['user_ids'][0]
    print(f"User ID: {sample_beat_saber_user}")
    print(f"Data shape: {authentication_data['beat_saber']['datasets'][sample_beat_saber_user].shape}")
    print(f"Class balance:")
    print(authentication_data['beat_saber']['datasets'][sample_beat_saber_user]['is_genuine'].value_counts())


    if TRAIN:
        # Create game authenticator instance
        authenticator = GameAuthenticator(authentication_data)
        # Train multiple models
        authenticator.train_multiple_models(apply_smote = True, filename = filename)

        # Save authenticator
        save_object(authenticator, output_filename)
    else:
        with open(output_filename, 'rb') as file:
            authenticator = pickle.load(file)

    # # Get performance metrics
    performance_metrics = authenticator.get_performance_summary()

    # # Show average performance by model type
    model_performance = performance_metrics.groupby(['Game_Type', 'Model_Type', 'Test_set']).mean().reset_index()
    print("Average model performance across all users:")
    display(model_performance[['Game_Type', 'Model_Type', 'Test_set', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC']])

    if SHAP:
        if SHAP_ALL:
            game_values, columns = authenticator.perform_shap(model_name='ET', filename=filename)
            shap_dict = {}
            shap_dict['values'] = game_values
            shap_dict['columns'] = columns
            save_object(shap_dict, shap_filename)
        elif SHAP_USER_BEAT:
            game_type_user = 'beat_saber'
            game_values, columns = authenticator.perform_shap_user(model_name='ET', filename=filename, user=USER, game=game_type_user)
            shap_dict = {}
            shap_dict['values'] = game_values
            shap_dict['columns'] = columns
        elif SHAP_USER_FORKLIFT:
            game_type_user = 'forklift_simulator'
            game_values, columns = authenticator.perform_shap_user(model_name='ET', filename=filename, user=USER, game=game_type_user)
            shap_dict = {}
            shap_dict['values'] = game_values
            shap_dict['columns'] = columns
        else:
            with open(shap_filename, 'rb') as file:
                shap_dict = pickle.load(file)

        feats = []
        columns = list(shap_dict['columns'])
        game_values = shap_dict['values']

        for col in columns:
            feat = '_'.join(col.split('_')[:-1])
            feats.append(feat)

        unique_feats = list(set(feats)) # sorting is necessary to have always the same order
        unique_feats.sort()

        if SHAP_USER_BEAT or SHAP_USER_FORKLIFT:
            game_types = [game_type_user]
        else:
            game_types = ['beat_saber', 'forklift_simulator']
        
        for game_type in game_types:
                for t in range(2):
                    values = game_values[game_type][t]
                    
                    summed_values = [] #list of shap values comulated per feature type, following the order in unique_feats
                    for feat in unique_feats:
                        ids = []
                        for id in range(len(columns)):
                            if feat in columns[id]:
                                ids.append(id)
                        summed_values.append(np.sum(values[ids]))
                        
                    idx_sort = np.array(summed_values).argsort()
                    
                    columns_sorted = np.array(unique_feats)[idx_sort[::-1]]
                    values_sorted = np.array(summed_values)[idx_sort[::-1]]
                    print(f'Game: {game_type} - test: {t}')
                    features = [a + ": " + str(b) for a,b in zip(columns_sorted, values_sorted)]
                    print(f'Game: {game_type_user} - test: {t}')
                    print(features[:30])


    if PERFORM_VOTING:
        # Implement voting
        if COMPUTE_VOTING:
            results = authenticator.perform_voting(voting_threshold=0.5, model_name="ET", filename=filename, output_filename=voting_results)
        else:
            with open(voting_results, 'rb') as file:
                results = pickle.load(file)

        authenticator.visualize_voting_results(results)

    print('-------------END-------------')
    

if __name__ == '__main__':
    main()
