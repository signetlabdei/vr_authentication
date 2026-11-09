import pandas as pd
import numpy as np
import pickle
import os

from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

INPUT_FILEPATH = '..\\data\\raw\\'
OUTPUT_FILEPATH = '..\\data\\processed_spoofing\\'
RESULTS_DIR = os.path.join('..\\results\\', 'authentication', 'spoofing_evaluation')
TIME_WINDOW = 5 # 5 second context window
TOTAL_TIME = 600 # 10 minutes of VR data per user
GAME_TYPE = 'fast'
SKIP_FRAME_SIZE = 10000 # Minimum size to consider a full frame

def perform_spoofing(filename_data,filename_windows,scale,norm_factors,filename_out):
    
    with open(filename_data, 'rb') as file:
        test_data_all = pickle.load(file) #contains the 7 subsets

    with open(filename_windows, 'rb') as file:
            windows = pickle.load(file) #contains the windows for the attack

    i=-1
    test_data_all_altered = []
    for test_data in test_data_all: # single test sample
        test_data = test_data.reset_index()
        victim = test_data['original_id'][0]
        i+=1 # subset index
        window_set = windows[i]
        j = -1 # attacker index
        for idx,data in test_data.iterrows():
            if data['is_genuine']==0:
                j+=1
                attacker = data['original_id']
                scale_values = scale.query("Attacker == @attacker & Victim == @victim") #extracts rows from scale df, note: @ allows to refer to variables
                
                window = window_set[j] # specific window for attacker idx
                altered_features = extract_features(victim,attacker,scale_values, window, norm_factors)

                for col in altered_features.columns.values:
                    test_data.loc[idx,col]=altered_features[col][0]

            for col in test_data.columns.values:
                if "PosY" in col and "Acc" not in col and "Vel" not in col:    
                    mean_value = norm_factors.query("id == @victim & col == @col")['val'] 
                    test_data.loc[idx,col] = test_data.loc[idx,col]/mean_value.item()

        test_data_all_altered.append(test_data)

    with open(filename_out, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(test_data_all_altered, outp, pickle.HIGHEST_PROTOCOL)
        

    
def alter_features(window_mf,scale_values):

    for id,data in window_mf.iterrows():
        
        window_mf.loc[id, 'HeadPosY'] = data['HeadPosY']*scale_values['HeadPosY'].values[0]
        NewX_head = data['HeadPosX']+scale_values['HeadPosX'].values[0]
        NewZ_head = data['HeadPosZ']+scale_values['HeadPosZ'].values[0]
        window_mf.loc[id,'HeadPosX'] = NewX_head
        window_mf.loc[id,'HeadPosZ'] = NewZ_head

        window_mf.loc[id,'LeftTouchPosY'] = data['LeftTouchPosY']*scale_values['LeftTouchPosY'].values[0]
        window_mf.loc[id,'RightTouchPosY'] = data['RightTouchPosY']*scale_values['RightTouchPosY'].values[0]

        #traslate attacker hands to victim hands so that heads are coincident
        
        NewX_left = data['LeftTouchPosX']+scale_values['HeadPosX'].values[0]
        NewZ_left = data['LeftTouchPosZ']+scale_values['HeadPosZ'].values[0]
        
        NewX_right = data['RightTouchPosX']+scale_values['HeadPosX'].values[0]
        NewZ_right = data['RightTouchPosZ']+scale_values['HeadPosZ'].values[0]
        
                    
        theta_left = scale_values['LeftRot'].values[0]
        
        R_left = np.array([
            [np.cos(theta_left), -np.sin(theta_left)],
            [np.sin(theta_left),  np.cos(theta_left)]
        ])

        H = np.array([NewX_head,NewZ_head])
        A_left = np.array([NewX_left,NewZ_left])
        
        A_left_rot = H + R_left @ (A_left - H)

        window_mf.loc[id,'LeftTouchPosX'] = A_left_rot[0]+scale_values['LeftTraslX'].values[0]
        window_mf.loc[id,'LeftTouchPosZ'] = A_left_rot[1]+scale_values['LeftTraslZ'].values[0]
                
        # repeat for right
                
        A_right = np.array([NewX_right,NewZ_right])
        
        theta_right = scale_values['RightRot'].values[0]

        R_right = np.array([
            [np.cos(theta_right), -np.sin(theta_right)],
            [np.sin(theta_right),  np.cos(theta_right)]
        ])

        A_right_rot = H + R_right @ (A_right - H)
        window_mf.loc[id,'RightTouchPosX'] = A_right_rot[0]+scale_values['RightTraslX'].values[0]
        window_mf.loc[id,'RightTouchPosZ'] = A_right_rot[1]+scale_values['RightTraslZ'].values[0]
    
    return window_mf

def angle_between(v1, v2):
    v1_u = v1 / np.linalg.norm(v1, axis=1)[:, np.newaxis]
    v2_u = v2 / np.linalg.norm(v2, axis=1)[:, np.newaxis]
    return np.arccos(np.clip(np.sum(v1_u * v2_u, axis=1), -1.0, 1.0))

def extract_features(victim,attacker,scale_values, window,norm_factors):
    # Create list of movement features, including position, velocity, and acceleration
    mf = pd.read_csv(INPUT_FILEPATH + 'group1_order1_user0/group1_order1_user0_fast_movement.csv')
    mov_values = list(mf.columns[5:-1]) 
    movement_feature_list = ['Pos_' + feature for feature in mov_values]
    movement_feature_list += ['Vel_' + feature for feature in mov_values]
    movement_feature_list += ['Acc_' + feature for feature in mov_values]
    movement_feature_list += ['Dist_LeftHead', 'Dist_RightHead', 'Dist_LeftRight', 'Angle_Left', 'Angle_Right', 'Angle_Head']

    

    stats = ['mean', 'std', 'min', 'max', 'q1', 'q2', 'q3']

    feature_list = [] 

    for stat in stats:
        feature_list += [feature + '_' + stat for feature in movement_feature_list]

    if attacker <=29:
        group_a = 1
    else:
        group_a =2
    
    if attacker <= 14 or (attacker>=30 and attacker<=44):
        order_a = 1
    else:
        order_a = 2

    attacker_user = int(attacker - (group_a - 1) * 30 - (order_a - 1) * 15)


    features = {}

    for feature in feature_list:
        features[feature] = []

    attacker_id = 'group' + str(group_a) + '_order' + str(order_a) + '_user' + str(attacker_user)
    
    mov_file_attacker = attacker_id + '/' + attacker_id + '_' + GAME_TYPE + '_movement.csv'


    print('Processing', mov_file_attacker)

    mf = pd.read_csv(INPUT_FILEPATH + mov_file_attacker)

    # We need to alter the features but only in the window of interest, i.e., the one selected during testing
    interval = int(window)
    window_mf = mf[(mf['time'] >= TIME_WINDOW * interval) & (mf['time'] < TIME_WINDOW * (interval + 1))]

    imf = alter_features(window_mf,scale_values)
    # Calculate derivatives
    for val in mov_values:
        imf['Pos_' + val] = imf[val]
        imf['Vel_' + val] = imf[val].diff(periods=1)
        imf['Acc_' + val] = imf[val].diff(periods=2)

    # Calculate distances and angles
    head_pos = imf[['HeadPosX', 'HeadPosZ']].values
    left_touch_pos = imf[['LeftTouchPosX', 'LeftTouchPosZ']].values
    right_touch_pos = imf[['RightTouchPosX', 'RightTouchPosZ']].values
    imf['Dist_LeftHead'] = np.linalg.norm(left_touch_pos - head_pos, axis=1)
    imf['Dist_RightHead'] = np.linalg.norm(right_touch_pos - head_pos, axis=1)
    imf['Dist_LeftRight'] = np.linalg.norm(left_touch_pos - right_touch_pos, axis=1)
    imf['Angle_Left'] = angle_between(left_touch_pos - head_pos, right_touch_pos - left_touch_pos)
    imf['Angle_Right'] = angle_between(right_touch_pos - head_pos, left_touch_pos - right_touch_pos)
    imf['Angle_Head'] = angle_between(head_pos - left_touch_pos, head_pos - right_touch_pos)


    for feature in movement_feature_list:
        features[feature + '_mean'].append(imf[feature].mean())
        features[feature + '_std'].append(imf[feature].std())
        features[feature + '_min'].append(imf[feature].min())
        features[feature + '_max'].append(imf[feature].max())
        features[feature + '_q1'].append(imf[feature].quantile(q=0.25))
        features[feature + '_q2'].append(imf[feature].quantile(q=0.5))
        features[feature + '_q3'].append(imf[feature].quantile(q=0.75))


    feature_frame = pd.DataFrame.from_dict(features)

    return feature_frame

def main():

    if GAME_TYPE == 'slow':
        for user_id in range(30,60):
            filename_data_test_only = RESULTS_DIR+'\\spoofing_data\\test_data_only_pre_attack_forklift_simulator'+ str(user_id) + '.pkl' #test only attackers data (extracted from genuine code)
            filename_windows_test_only = RESULTS_DIR+'\\spoofing_data\\test_only_window_ids_pre_attack_forklift_simulator' + str(user_id) + '.pkl' # windows selected for the test (extracted from genuine code)
            filename_data_test_trained = RESULTS_DIR+'\\spoofing_data\\test_data_trained_pre_attack_forklift_simulator' + str(user_id) + '.pkl' #trained attackers data (extracted from genuine code)
            filename_windows_test_trained = RESULTS_DIR+'\\spoofing_data\\test_trained_window_ids_pre_attack_forklift_simulator' + str(user_id) + '.pkl' # windows selected for the test (extracted from genuine code)
            scale = pd.read_csv('..\\data\\processed_spoofing\\slow_scale_factors.csv').fillna(0) # <- produces by spoofing_parameters.py
            norm_factors = pd.read_csv('norm_factor_forklift.csv') # genuine height normalization factors to mimic a real system which estimates this parameter during training (extracted from genuine code)
            filename_out = RESULTS_DIR+'\\spoofing_data\\test_data_post_attack_forklift_simulator' + str(user_id)
            perform_spoofing(filename_data_test_only,filename_windows_test_only, scale,norm_factors,filename_out+'_test_only.pkl')
            perform_spoofing(filename_data_test_trained,filename_windows_test_trained, scale,norm_factors,filename_out+'_test_trained.pkl')
    else: # repeat the same for the fast game
        for user_id in range(30):
            filename_data_test_only = RESULTS_DIR+'\\spoofing_data\\test_data_only_pre_attack_beat_saber'+ str(user_id) + '.pkl' 
            filename_windows_test_only = RESULTS_DIR+'\\spoofing_data\\test_only_window_ids_pre_attack_beat_saber' + str(user_id) + '.pkl'
            filename_data_test_trained = RESULTS_DIR+'\\spoofing_data\\test_data_trained_pre_attack_beat_saber' + str(user_id) + '.pkl' 
            filename_windows_test_trained = RESULTS_DIR+'\\spoofing_data\\test_trained_window_ids_pre_attack_beat_saber' + str(user_id) + '.pkl'
            scale = pd.read_csv('..\\data\\processed_spoofing\\fast_scale_factors.csv').fillna(0)
            norm_factors = pd.read_csv('norm_factor_beat.csv')
            filename_out = RESULTS_DIR+'\\spoofing_data\\test_data_post_attack_beat_saber' + str(user_id)
            perform_spoofing(filename_data_test_only,filename_windows_test_only, scale,norm_factors,filename_out+'_test_only.pkl')
            perform_spoofing(filename_data_test_trained,filename_windows_test_trained, scale,norm_factors,filename_out+'_test_trained.pkl')

     

if __name__=='__main__':
     main()