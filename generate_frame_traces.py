import pandas as pd
import numpy as np
from pathlib import Path

INPUT_FILEPATH = "./data/raw/"

for group in [1,2]:
    for user in range(15):
        for order in [1,2]:
            for game in ['fast', 'slow']:
                userid = 'group' + str(group) + '_order' + str(order) + '_user' + str(user)
                traffic_file = INPUT_FILEPATH + userid + '/' + userid + '_' + game + '_traffic.csv'
                frame_file = INPUT_FILEPATH + userid + '/' + userid + '_' + game + '_frames.csv'
                print(userid, game)
                trace_data = pd.read_csv(traffic_file)

                # set a threshold to filter out the smallest packets
                size_th = 1e3  # [B]
                theoretical_fps = 72  # [fps]
                theoretical_ts = 1 / theoretical_fps  # [s]

                # keep only relevant downlink data
                dl_data = trace_data[trace_data["direction"] == "DL"]
                dl_data = dl_data[dl_data["size"] > size_th]

                # extract packet time and size
                time_pkt = dl_data["time"].values
                size_pkt = dl_data["size"].values

                # compute the inter packet interval times
                ipi = np.diff(time_pkt)

                # group the packets that are closer than frame_t_thr_s into the same frame
                frame_t_thr_s = 5e-3  # [s]
                pkt_grp = ipi < frame_t_thr_s

                frame_idx = 0
                frame_size = []
                frame_time = []
                frame_pkts = []
                tmp_size_b = size_pkt[0]
                tmp_time_pkt = time_pkt[0]
                pkt_counter = 0
                frame_ids = [0]
                frame_counter = 0
                for ipi_idx, _ in enumerate(pkt_grp):
                    if pkt_grp[ipi_idx]:
                        if pkt_counter == 0:
                            tmp_time_pkt = time_pkt[ipi_idx]
                        pkt_counter += 1
                        tmp_size_b += size_pkt[ipi_idx + 1]
                    else:
                        frame_pkts.append(pkt_counter)
                        frame_size.append(tmp_size_b)
                        frame_time.append(tmp_time_pkt)
                        tmp_size_b = size_pkt[ipi_idx + 1]
                        pkt_counter = 1
                        tmp_time_pkt = time_pkt[ipi_idx + 1]
                        frame_counter = frame_counter + 1
                    frame_ids.append(frame_counter)

                frame_size = np.array(frame_size)
                frame_time = np.array(frame_time)
                frame_ids = np.array(frame_ids)
                frame_pkts = np.array(frame_pkts)

                # if the time between frames is larger than the theoretical sampling time (1/FPS)
                # fill the gaps with zero-size frames
                ifi = np.diff(frame_time)
                gaps_ids = ifi >= 1.5 / theoretical_fps

                frame_time_fill = []
                frame_size_fill = []
                frame_pkts_fill = []
                for frame_idx, _ in enumerate(gaps_ids):
                    frame_time_fill.append(frame_time[frame_idx])
                    frame_size_fill.append(frame_size[frame_idx])
                    frame_pkts_fill.append(frame_pkts[frame_idx])
                    if gaps_ids[frame_idx]:
                        frame_time_0 = frame_time[frame_idx]
                        frame_time_1 = frame_time[frame_idx + 1]
                        gap_duration = frame_time_1 - frame_time_0
                        n_frames_gap = gap_duration // theoretical_ts + 1

                        for i in np.arange(1, int(n_frames_gap)):
                            new_frame_time = frame_time_0 + i * theoretical_ts
                            if new_frame_time >= frame_time_1 - theoretical_ts * 0.5:
                                break
                            frame_time_fill.append(new_frame_time)
                            frame_size_fill.append(0)
                            frame_pkts_fill.append(0)

                frame_time_fill = np.array(frame_time_fill)
                frame_size_fill = np.array(frame_size_fill)
                frame_pkts_fill = np.array(frame_pkts_fill)

                frame_data_fill = {
                    "time": frame_time_fill,
                    "size": frame_size_fill,
                    "pkts": frame_pkts_fill,
                }
                frame_data_fill_df = pd.DataFrame(frame_data_fill)
                frame_data_fill_df.to_csv(frame_file, index=False)
