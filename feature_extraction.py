import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def angle_between(v1, v2):
    v1_u = v1 / np.linalg.norm(v1, axis=1)[:, np.newaxis]
    v2_u = v2 / np.linalg.norm(v2, axis=1)[:, np.newaxis]
    return np.arccos(np.clip(np.sum(v1_u * v2_u, axis=1), -1.0, 1.0))

INPUT_FILEPATH = './data/raw/'
OUTPUT_FILEPATH = './data/processed/'
TIME_WINDOW = 5 # 5 second context window
TOTAL_TIME = 600 # 10 minutes of VR data per user
GAME_TYPE = 'slow'
SKIP_FRAME_SIZE = 10000 # Minimum size to consider a full frame

# Create list of movement features, including position, velocity, and acceleration
mf = pd.read_csv(INPUT_FILEPATH + 'group1_order1_user0/group1_order1_user0_fast_movement.csv')
mov_values = list(mf.columns[7:-1])
movement_feature_list = ['Pos_' + feature for feature in mov_values]
movement_feature_list += ['Vel_' + feature for feature in mov_values]
movement_feature_list += ['Acc_' + feature for feature in mov_values]
movement_feature_list += ['Dist_LeftHead', 'Dist_RightHead', 'Dist_LeftRight', 'Angle_Left', 'Angle_Right', 'Angle_Head']

# Create traffic features
traffic_feature_list = ['Traffic_DL_Packet_Size', 'Traffic_DL_Frame_Size']

stats = ['mean', 'std', 'min', 'max', 'q1', 'q2', 'q3']

feature_list = ['time', 'ID', 'Traffic_Total_Data', 'Traffic_Missing_Frames', 'Traffic_UL_Packets', 'Traffic_DL_Packets']

for stat in stats:
    feature_list += [feature + '_' + stat for feature in movement_feature_list]
    feature_list += [feature + '_' + stat for feature in traffic_feature_list]

for group in [1,2]:
    for user in range(15):
        for order in [1,2]:
            features = {}

            for feature in feature_list:
                features[feature] = []

            userid = 'group' + str(group) + '_order' + str(order) + '_user' + str(user)
            id_number = (group - 1) * 30 + (order - 1) * 15 + user
            traffic_file = userid + '/' + userid + '_' + GAME_TYPE + '_traffic.csv'
            frame_file = userid + '/' + userid + '_' + GAME_TYPE + '_frames.csv'
            mov_file = userid + '/' + userid + '_' + GAME_TYPE + '_movement.csv'

            output_folder = Path(OUTPUT_FILEPATH).joinpath(userid + '/')
            if not output_folder.exists():
                output_folder.mkdir(parents=True, exist_ok=True)

            print('Processing', userid)

            tf = pd.read_csv(INPUT_FILEPATH + traffic_file)
            ff = pd.read_csv(INPUT_FILEPATH + frame_file)
            mf = pd.read_csv(INPUT_FILEPATH + mov_file)

            # Calculate derivatives
            for val in mov_values:
                mf['Pos_' + val] = mf[val]
                mf['Vel_' + val] = mf[val].diff(periods=1)
                mf['Acc_' + val] = mf[val].diff(periods=2)

            # Calculate distances and angles
            head_pos = mf[['HeadPosX', 'HeadPosY']].values
            left_touch_pos = mf[['LeftTouchPosX', 'LeftTouchPosY']].values
            right_touch_pos = mf[['RightTouchPosX', 'RightTouchPosY']].values
            mf['Dist_LeftHead'] = np.linalg.norm(left_touch_pos - head_pos, axis=1)
            mf['Dist_RightHead'] = np.linalg.norm(right_touch_pos - head_pos, axis=1)
            mf['Dist_LeftRight'] = np.linalg.norm(left_touch_pos - right_touch_pos, axis=1)
            mf['Angle_Left'] = angle_between(left_touch_pos - head_pos, right_touch_pos - left_touch_pos)
            mf['Angle_Right'] = angle_between(right_touch_pos - head_pos, left_touch_pos - right_touch_pos)
            mf['Angle_Head'] = angle_between(head_pos - left_touch_pos, head_pos - right_touch_pos)

            for interval in range(int(np.ceil(TOTAL_TIME / TIME_WINDOW))):
                imf = mf[(mf['time'] >= TIME_WINDOW * interval) & (mf['time'] < TIME_WINDOW * (interval + 1))]
                itf = tf[(tf['time'] >= TIME_WINDOW * interval) & (tf['time'] < TIME_WINDOW * (interval + 1))]
                iff = ff[(ff['time'] >= TIME_WINDOW * interval) & (ff['time'] < TIME_WINDOW * (interval + 1))]

                itf_dlsize = itf[itf['direction'] == 'DL']['size']

                features['time'].append(TIME_WINDOW * interval / 60)
                for feature in movement_feature_list:
                    features[feature + '_mean'].append(imf[feature].mean())
                    features[feature + '_std'].append(imf[feature].std())
                    features[feature + '_min'].append(imf[feature].min())
                    features[feature + '_max'].append(imf[feature].max())
                    features[feature + '_q1'].append(imf[feature].quantile(q=0.25))
                    features[feature + '_q2'].append(imf[feature].quantile(q=0.5))
                    features[feature + '_q3'].append(imf[feature].quantile(q=0.75))

                feature = 'Traffic_DL_Packet_Size'
                features[feature + '_mean'].append(itf_dlsize.mean())
                features[feature + '_std'].append(itf_dlsize.std())
                features[feature + '_min'].append(itf_dlsize.min())
                features[feature + '_max'].append(itf_dlsize.max())
                features[feature + '_q1'].append(itf_dlsize.quantile(q=0.25))
                features[feature + '_q2'].append(itf_dlsize.quantile(q=0.5))
                features[feature + '_q3'].append(itf_dlsize.quantile(q=0.75))

                feature = 'Traffic_DL_Frame_Size'
                iff_dlsize = iff['size']
                features[feature + '_mean'].append(iff_dlsize.mean())
                features[feature + '_std'].append(iff_dlsize.std())
                features[feature + '_min'].append(iff_dlsize.min())
                features[feature + '_max'].append(iff_dlsize.max())
                features[feature + '_q1'].append(iff_dlsize.quantile(q=0.25))
                features[feature + '_q2'].append(iff_dlsize.quantile(q=0.5))
                features[feature + '_q3'].append(iff_dlsize.quantile(q=0.75))

                features['Traffic_Total_Data'].append(itf_dlsize.sum())
                features['Traffic_DL_Packets'].append(itf[itf['direction'] == 'DL'].shape[0])
                features['Traffic_UL_Packets'].append(itf[itf['direction'] == 'UL'].shape[0])
                features['Traffic_Missing_Frames'].append(iff_dlsize[iff_dlsize < 10000].shape[0])
                features['ID'].append(id_number)

            feature_frame = pd.DataFrame.from_dict(features)
            feature_frame.to_csv(OUTPUT_FILEPATH + userid + '/' + GAME_TYPE + '_features.csv')
