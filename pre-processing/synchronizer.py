import pandas as pd
import numpy as np

INPUT_FILEPATH = "./data/raw/"

for group in [1,2]:
    for user in range(15):
        for order in [1,2]:
            for game in ['fast', 'slow']:
                userid = 'group' + str(group) + '_order' + str(order) + '_user' + str(user)
                traffic_file = INPUT_FILEPATH + userid + '/' + userid + '_' + game + '_traffic.csv'
                mov_file = INPUT_FILEPATH + userid + '/' + userid + '_' + game + '_movement.csv'

                tf = pd.read_csv(traffic_file)
                mf = pd.read_csv(mov_file)

                t0m = np.min(mf['time'])
                t0f = np.min(tf['time'])

                print(user, game, t0m, t0f)

                t0 = np.max([t0f, t0m])

                tf = tf[tf['time'] >= t0]
                mf = mf[mf['time'] >= t0]
                mf['time'] -= t0
                tf['time'] -= t0

                t0m = np.min(mf['time'])
                t0f = np.min(tf['time'])
                print(user, game, t0m, t0f)
                tf.to_csv(traffic_file)
                mf.to_csv(mov_file)
