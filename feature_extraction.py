import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

INPUT_FILEPATH = './data/raw/'
OUTPUT_FILEPATH = './data/processed/'
TIME_WINDOW = 5 # 5 second context window
TOTAL_TIME = 600 # 10 minutes of VR data per user
GAME_TYPE = 'slow'

# Create list of movement features, including position, velocity, and acceleration
mf = pd.read_csv(INPUT_FILEPATH + 'group1_order1_user0/group1_order1_user0_fast_movement.csv')
mov_values = list(mf.columns[7:-1])
feature_type_list = ['Pos_' + feature for feature in mov_values]
feature_type_list += ['Vel_' + feature for feature in mov_values]
feature_type_list += ['Acc_' + feature for feature in mov_values]
feature_type_list += ['Dist_LeftHead', 'Dist_RightHead', 'Dist_LeftRight', 'Angle_LeftHead', 'Angle_RightHead', 'Angle_LeftRight']

# Create traffic features
feature_type_list += ['Traffic_DL_Size', 'Traffic_UL_Packets', 'Traffic_DL_Packets']

stats = ['min', 'max', 'mean', 'std', 'q1', 'q2', 'q3']

feature_list = ['time', 'ID', 'Traffic_Total_Data']

for stat in stats:
    feature_list += [feature + '_' + stat for feature in feature_type_list]

features = {}

for feature in feature_list:
    features[feature] = []



for group in [1,2]:
    for user in range(1):
        for order in [1,2]:
            userid = 'group' + str(group) + '_order' + str(order) + '_user' + str(user)
            traffic_file = userid + '/' + userid + '_' + GAME_TYPE + '_traffic.csv'
            mov_file = userid + '/' + userid + '_' + GAME_TYPE + '_movement.csv'

            print('Processing', userid)

            tf = pd.read_csv(INPUT_FILEPATH + traffic_file)
            mf = pd.read_csv(INPUT_FILEPATH + mov_file)

            for interval in range(int(np.ceil(TOTAL_TIME / TIME_WINDOW))):
                imf = mf[(mf['time'] >= TIME_WINDOW * interval) & (mf['time'] < TIME_WINDOW * (interval + 1))]
                itf = tf[(tf['time'] >= TIME_WINDOW * interval) & (tf['time'] < TIME_WINDOW * (interval + 1))]
                print(imf['time'])

