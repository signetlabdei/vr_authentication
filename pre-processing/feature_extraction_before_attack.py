import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def angle_between(v1, v2):
    v1_u = v1 / np.linalg.norm(v1, axis=1)[:, np.newaxis]
    v2_u = v2 / np.linalg.norm(v2, axis=1)[:, np.newaxis]
    return np.arccos(np.clip(np.sum(v1_u * v2_u, axis=1), -1.0, 1.0))

INPUT_FILEPATH = '..\\data\\raw\\'
OUTPUT_FILEPATH = '..\\data\\processed_spoofing\\'
TRAIN_TIME = 480 # 8 minutes of VR data are used for training
GAME_TYPE = 'fast'

# Create list of movement data

mov_values = ['time', 'HeadPosX', 'HeadPosY', 'HeadPosZ','RightTouchPosX', 'RightTouchPosY', 'RightTouchPosZ','LeftTouchPosX', 'LeftTouchPosY', 'LeftTouchPosZ']
user_tab={'UserId':[], 'HeadPosX':[], 'HeadPosY':[], 'HeadPosZ':[],'RightTouchPosX':[], 'RightTouchPosY':[], 'RightTouchPosZ':[],'LeftTouchPosX':[], 'LeftTouchPosY':[], 'LeftTouchPosZ':[]}
dict_keys = list(user_tab.keys())

for group in [1,2]:
    for user in range(15):
        for order in [1,2]:

            userid = 'group' + str(group) + '_order' + str(order) + '_user' + str(user)
            id_number = (group - 1) * 30 + (order - 1) * 15 + user
            mov_file = userid + '\\' + userid + '_' + GAME_TYPE + '_movement.csv'


            print('Processing', userid)

            mf = pd.read_csv(INPUT_FILEPATH + mov_file)

            mf_feat = mf[mov_values]

            imf = mf_feat[(mf_feat['time'] <= TRAIN_TIME)] # Select only training samples

            single_user = (imf.mean()[1:]) # Average over time and removal of the time column

            for key in dict_keys[1:]:
                user_tab[key].append(single_user[key])

            user_tab['UserId'].append(id_number)

            

user_frame = pd.DataFrame.from_dict(user_tab)


user_frame.to_csv(OUTPUT_FILEPATH + '\\' + GAME_TYPE + '_means.csv',index=False)
