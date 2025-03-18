import pandas as pd
import utils
import trial_analysis
import saccade_analysis
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm     
"""
Viewing distance = 19 cm    
Monitor : 1920X1080 
32 inches wide (24inch x 13.5inches)
Locations:  x_start, y_start, x_end, y_end
https://nevada.app.box.com/folder/302553040721
   x_start  y_start  x_end  y_end  x_center  y_center      x_cm      y_cm    azimuth  elevation  azimuth_radius  elevation_radius
0     1277      410   1453    670    1365.0     540.0  12.85875   0.00000  37.390421   0.000000        7.552641         12.398149
1     1075      761   1251   1021    1163.0     891.0   6.44525 -11.14425  19.252858 -32.689633        8.177176         11.432710
2      670      761    846   1021     758.0     891.0  -6.41350 -11.14425 -19.159778 -32.689633        8.179406         11.432710
3      467      410    643    670     555.0     540.0 -12.85875   0.00000 -37.390421   0.000000        7.552641         12.398149
4      670       59    846    319     758.0     189.0  -6.41350  11.14425 -19.159778  32.689633        8.179406         11.432710
5     1075       59   1251    319    1163.0     189.0   6.44525  11.14425  19.252858  32.689633        8.177176         11.432710
"""
def main():
    trial_df, event_df = utils.read_trials_and_event()
    gx = pd.read_csv('gx.csv',header=None).T.drop(columns=[0])
    gy = pd.read_csv('gy.csv',header=None).T.drop(columns=[0])
    time_eye =   pd.read_csv('time.csv',header=None).T
    df = pd.DataFrame({
        'x': gx[1].values,
        'y': gy[1].values,
        'time': time_eye[0].values
    })
    save_time = df['time']
    eyelink = utils.load_eyelink(df)
    eyelink['azimuth'], eyelink['elevation'], quality_metrics, target_df = utils.filter_eyetracking_data(eyelink['azimuth'].values,eyelink['elevation'].values,eyelink.index)
    eyelink.index = save_time
    eyelink = utils.set_norm(eyelink)
    trials_w_times = trial_analysis.extract_time_stamps_for_each_trial(event_df)

    #saccade detection
    saccade_df = saccade_analysis.detect_saccades_eyelink(event_df)
    saccade_offsets_ts = saccade_df['entime']
    saccade_onset_ts = saccade_df['sttime']
    # eyelink = saccade_analysis.derivatives(eyelink)
    # pt_thresh = saccade_analysis.find_velocity_peaks(eyelink)
    # saccade_onsets_ts, saccade_offsets_ts = saccade_analysis.detect_saccades(eyelink,pt_thresh)

    target_df['max_radius'] = target_df[['azimuth_radius', 'elevation_radius']].max(axis=1)
    trial_count = 0
    dict_save = {}
    for idx in tqdm(range(len(trial_df))):
        print(f'current trial: {idx+1}')
        trial_start = trials_w_times['sttime'].iloc[idx]
        trial_end = trial_start + (trial_df['RT'].iloc[idx] * 1000)
        current_target = trial_df['TargetPos'].iloc[idx]
        curr_target_pos = target_df.iloc[current_target-1]
        
        target_center = curr_target_pos[['azimuth', 'elevation']].values
        target_radius = curr_target_pos['max_radius']
        
        filtered_data = eyelink[(eyelink.index >= trial_start) & (eyelink.index <= trial_end)]
        
        utils.plot_saccade_landings(idx,target_center,target_radius,target_df,current_target,saccade_offsets_ts,trial_start,trial_end,filtered_data)
        num_saccades_per_trial = 0
        number_saccades_target, number_saccades_distractor = 0, 0
        saccade_vel_list_target,saccade_amp_list_target = [], []
        saccade_vel_list_dist,saccade_amp_list_dist = [], []
        for offset_ts, onset_ts in zip(saccade_offsets_ts,saccade_onset_ts):
            if trial_start <= offset_ts <= trial_end:
                num_saccades_per_trial += 1
                #get the eye position at the saccade offset timestamp
                eye_pos = filtered_data.loc[offset_ts, ['azimuth', 'elevation']].values
                eye_signal = filtered_data.loc[onset_ts:offset_ts, ['azimuth', 'elevation']].abs()
                
                if eye_signal.isna().all().all():
                    #all the eye data are nans due to noise or loss of eye signal
                    saccade_vel_list_target.append(np.nan)
                    saccade_amp_list_target.append(np.nan)
                    saccade_vel_list_dist.append(np.nan)
                    saccade_amp_list_dist.append(np.nan)
                else:
                    if eye_signal.isna().any().any():
                        eye_signal = filtered_data.loc[onset_ts:offset_ts, ['azimuth', 'elevation']].dropna().abs()

                    amplitude_euclid = np.sqrt(np.sum((eye_signal.values[-1] - eye_signal.values[0])**2))
                    total_time = (eye_signal.index[-1] - eye_signal.index[0] ) / 1000
                    velocity = amplitude_euclid / total_time

                    #calculate distance from the eye position to the target center
                    distance_to_target = np.linalg.norm(eye_pos - target_center)
                    if distance_to_target <= target_radius:
                        number_saccades_target += 1
                        
                        saccade_amp_list_target.append(amplitude_euclid)
                        saccade_vel_list_target.append(velocity)
                        print(f"Saccade offset at {offset_ts} landed in the target ROI")

                    #check against all distractors 
                    for distractor_idx, distractor_row in target_df.iterrows():
                        #skip the target ROI
                        if distractor_idx == current_target - 1:
                            continue
                        
                        distractor_center = distractor_row[['azimuth', 'elevation']].values
                        distractor_radius = distractor_row['max_radius']
                        distance_to_distractor = np.linalg.norm(eye_pos - distractor_center)
                        if distance_to_distractor <= distractor_radius:
                            number_saccades_distractor += 1
                            saccade_amp_list_dist.append(amplitude_euclid)
                            saccade_vel_list_dist.append(velocity)
                            print(f"Saccade offset at {offset_ts} landed in a distractor ROI")
        key = f'{idx+1}'
        if key not in dict_save:
            dict_save[key] = {}
        dict_save[f'{idx+1}']["saccade_vel_target"] = np.nanmean(saccade_vel_list_target)
        dict_save[f'{idx+1}']["saccade_amp_target"] = np.nanmean(saccade_amp_list_target)
        dict_save[f'{idx+1}']["saccade_vel_dist"] = np.nanmean(saccade_vel_list_dist)
        dict_save[f'{idx+1}']["saccade_amp_dist"] = np.nanmean(saccade_amp_list_dist)
        dict_save[f'{idx+1}']["num_sacc_dist"] = number_saccades_distractor
        dict_save[f'{idx+1}']["num_sacc_target"] = number_saccades_target
        # print(f'Number of saccades during trial: {num_saccades_per_trial}')
        # print(f'number of saccades (dist): {number_saccades_distractor}')
        # print(f'number of saccades (target): {number_saccades_target}')
        
    df_final = pd.DataFrame.from_dict(dict_save, orient='index')
    df_final.to_csv('test_participant.csv',index=False)

if __name__ == "__main__":
    main()