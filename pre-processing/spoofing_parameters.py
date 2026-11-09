import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def plane_rotation(A, V, H): #take as input np arrays of attacker and victim hands and head position
    
    
    # compute vectors with respect to head and normalize
    HA = A - H
    HV = V - H

    HA = HA / np.linalg.norm(HA)
    HV = HV / np.linalg.norm(HV)
    
    dot = np.dot(HA, HV) # for angle width
    cross = HA[0]*HV[1] - HA[1]*HV[0]  # for angle sign (positive rotations to the right)
    
    theta = np.arctan2(cross, dot)  # randiants
    return theta

INPUT_FILEPATH = '..\\data\\processed_spoofing\\'
OUTPUT_FILEPATH = '..\\data\\processed_spoofing\\'
GAME_TYPE = 'slow'

# Create list of movement features, including position, velocity, and acceleration


mf = pd.read_csv(INPUT_FILEPATH + GAME_TYPE + '_means.csv') # -> from feature_extracttion_per_attack.py
users = list(mf['UserId'])
mov_values = ['UserId', 'HeadPosX', 'HeadPosY', 'HeadPosZ','RightTouchPosX', 'RightTouchPosY', 'RightTouchPosZ','LeftTouchPosX', 'LeftTouchPosY', 'LeftTouchPosZ']
scale_dict = {'Attacker':[], 'Victim':[], 'HeadPosX':[], 'HeadPosY':[], 'HeadPosZ':[],'RightRot':[], 'RightTraslX':[], 'RightTraslZ':[], 'RightTouchPosY':[],'LeftRot':[], 'LeftTraslX':[], 'LeftTraslZ':[], 'LeftTouchPosY':[]}

for user in users:
    victims = [x for x in users if x != user]
    
    attacker = mf[mf['UserId']==user]
    for victim in victims:
        scale_dict['Attacker'].append(user)
        scale_dict['Victim'].append(victim)
        victim = mf[mf['UserId']==victim]

        scale_dict['HeadPosY'].append(victim['HeadPosY'].values[0]/attacker['HeadPosY'].values[0]) # scale factor head vertical
        DeltaX_head = victim['HeadPosX'].values[0]-attacker['HeadPosX'].values[0] # scale factor head X
        DeltaZ_head = victim['HeadPosZ'].values[0]-attacker['HeadPosZ'].values[0] ## scale factor head Y
        scale_dict['HeadPosX'].append(DeltaX_head)
        scale_dict['HeadPosZ'].append(DeltaZ_head)

        scale_dict['LeftTouchPosY'].append(victim['LeftTouchPosY'].values[0]/attacker['LeftTouchPosY'].values[0]) # scale factor left vertical
        scale_dict['RightTouchPosY'].append(victim['RightTouchPosY'].values[0]/attacker['RightTouchPosY'].values[0]) #scale factor right vertical

        #traslate attacker hands to victim hands so that heads are coincident

        NewX_left = attacker['LeftTouchPosX'].values[0]+DeltaX_head
        NewZ_left = attacker['LeftTouchPosZ'].values[0]+DeltaZ_head

        NewX_right = attacker['RightTouchPosX'].values[0]+DeltaX_head
        NewZ_right = attacker['RightTouchPosZ'].values[0]+DeltaZ_head

        # prepare vectors for rotation from attacker to victim and compute angle
        H = np.array([victim['HeadPosX'].values[0],victim['HeadPosZ'].values[0]])

        A_left = np.array([NewX_left,NewZ_left])
        V_left = np.array([victim['LeftTouchPosX'].values[0],victim['LeftTouchPosZ'].values[0]])

        theta_left = plane_rotation(A_left, V_left, H)
        scale_dict['LeftRot'].append(theta_left)

        R_left = np.array([
            [np.cos(theta_left), -np.sin(theta_left)],
            [np.sin(theta_left),  np.cos(theta_left)]
        ])

        A_left_rot = H + R_left @ (A_left - H)
        scale_dict['LeftTraslX'].append(victim['LeftTouchPosX'].values[0]-A_left_rot[0])
        scale_dict['LeftTraslZ'].append(victim['LeftTouchPosZ'].values[0]-A_left_rot[1])

        # repeat for right
        
        A_right = np.array([NewX_right,NewZ_right])
        V_right = np.array([victim['RightTouchPosX'].values[0],victim['RightTouchPosZ'].values[0]])
        theta_right = plane_rotation(A_right, V_right, H)
        scale_dict['RightRot'].append(theta_right)

        R_right = np.array([
            [np.cos(theta_right), -np.sin(theta_right)],
            [np.sin(theta_right),  np.cos(theta_right)]
        ])

        A_right_rot = H + R_right @ (A_right - H)
        scale_dict['RightTraslX'].append(victim['RightTouchPosX'].values[0]-A_right_rot[0])
        scale_dict['RightTraslZ'].append(victim['RightTouchPosZ'].values[0]-A_right_rot[1])


user_frame = pd.DataFrame.from_dict(scale_dict)


user_frame.to_csv(OUTPUT_FILEPATH + '\\' + GAME_TYPE + '_scale_factors.csv',index=False)
