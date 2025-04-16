
def dwell_times(target_center, target_radius, target_df, current_target, trial_start, trial_end, filtered_data):
    # data is in the trial window
    trial_data = filtered_data.loc[trial_start:trial_end]

    # timers
    dwell_time_target = 0.0
    dwell_time_distractor = 0.0

    # timestamps as array
    timestamps = trial_data.index.to_numpy()

    #iterate over gaze points in the trial
    for i in range(len(trial_data) - 1):
        az, el = trial_data.iloc[i][['azimuth', 'elevation']]
        t1, t2 = timestamps[i], timestamps[i + 1]
        #how long the gaze was at this location
        duration = t2 - t1  

        # Check if in target ROI
        if (
            abs(az - target_center[0]) <= target_radius and
            abs(el - target_center[1]) <= target_radius
        ):
            dwell_time_target += duration
        else:
            # Check distractors
            for distractor_idx, distractor_row in target_df.iterrows():
                if distractor_idx == current_target - 1:
                    continue

                dx, dy = distractor_row[['azimuth', 'elevation']]
                dr = distractor_row['max_radius']

                if (
                    abs(az - dx) <= dr and
                    abs(el - dy) <= dr
                ):
                    dwell_time_distractor += duration
                    # count once per time segment
                    break  

    return float(dwell_time_target), float(dwell_time_distractor)
