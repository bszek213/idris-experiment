import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def gaze_to_azel(x, y, distance):
    azimuth = np.arctan2(x, distance)
    elevation = np.arctan2(y, distance)
    azimuth_deg = np.degrees(azimuth)
    elevation_deg = np.degrees(elevation)
    return azimuth_deg, elevation_deg

def gaze_to_azel_arctan(x, y, distance):
    azimuth = np.arctan(x / distance)
    azimuth = np.where(x < 0, azimuth + np.pi, azimuth)
    azimuth = np.where((x >= 0) & (distance < 0), azimuth + np.pi, azimuth)
    azimuth = np.where(azimuth > np.pi, azimuth - 2*np.pi, azimuth)
    elevation = np.arctan(y / np.sqrt(x**2 + distance**2))
    azimuth_deg = np.degrees(azimuth)
    elevation_deg = np.degrees(elevation)
    azimuth_deg = azimuth_deg - np.mean(azimuth_deg)
    elevation_deg = elevation_deg - np.mean(elevation_deg)
    return azimuth_deg, elevation_deg

def load_eyelink(
    eyelink, screen_size=(0.6096, 0.3429), resolution=(1920, 1080), start_time=None 
):
    distance = 0.19
    eyelink.index = eyelink['time']
    eyelink.drop(columns=['time'], inplace=True)

    # convert timestamps to time since start
    ts_eyelink = eyelink.index - eyelink.index[0]
    if start_time is not None:
        eyelink = eyelink.set_index(
            pd.to_datetime(start_time) + pd.to_timedelta(ts_eyelink, unit="ms")
        )
    else:
        eyelink = eyelink.set_index(pd.to_timedelta(ts_eyelink, unit="ms"))

    # calculate gaze positions in meters relative to screen center
    # we need to multiply with -1 because the EyeLink screen coordinates are x right, y up
    # but Optitrack coordinates are x left, y up
    width_factor = screen_size[0] / resolution[0]  # pixel width
    eyelink['x'] = (
        eyelink['x'] * width_factor - (screen_size[0] / 2)
    )
    height_factor = screen_size[1] / resolution[1]  # pixel height
    eyelink['y'] = (
        eyelink['y'] * height_factor - (screen_size[1] / 2)
    )

    # set depth to zero
    eyelink["z"] = 0

    # sort columns by name
    eyelink = eyelink.reindex(sorted(eyelink.columns), axis=1)

    # drop NaNs
    #eyelink = eyelink.dropna()
    #convert to degrees
    eyelink['azimuth'], eyelink['elevation'] = gaze_to_azel(eyelink['x'], eyelink['y'], distance)


    return eyelink

def filter_eyetracking_data(azimuth, elevation, time_index, sampling_rate=500,
                            velocity_threshold=300, signal_loss_value=88,
                            tolerance=1e-3, max_gap_length=75):
    """
    Filter eye tracking data to remove signal loss and noise using research-backed methods.
    Optimized for 500 Hz sampling rate.
    """
    # Convert data to numpy arrays
    az = np.array(azimuth)
    el = np.array(elevation)
    
    # Create mask for signal loss with tolerance for floating-point comparisons
    signal_loss_mask = (az >= signal_loss_value) | (el >= signal_loss_value)

    # Replace signal loss values with NaN (or any placeholder you prefer)
    az[signal_loss_mask] = np.nan
    el[signal_loss_mask] = np.nan
    
    # Calculate sample-to-sample velocity
    time_diff = (time_index[1:] - time_index[:-1]).total_seconds()  # Time differences in seconds
    az_vel = np.diff(az) / time_diff  # azimuth velocity in deg/s
    el_vel = np.diff(el) / time_diff  # elevation velocity in deg/s
    total_vel = np.sqrt(az_vel**2 + el_vel**2)
    
    # Pad velocity array to match original length
    total_vel = np.pad(total_vel, (0, 1), mode='edge')
    
    # Create mask for high velocity samples
    velocity_mask = total_vel > velocity_threshold
    
    # Combine masks to identify invalid samples
    invalid_samples = signal_loss_mask | velocity_mask
    
    # Interpolate gaps shorter than max_gap_length
    max_samples_gap = int(max_gap_length * sampling_rate / 1000)  # ~37-38 samples at 500 Hz
    
    # Find sequences of invalid samples
    invalid_runs = np.where(invalid_samples)[0]
    if len(invalid_runs) > 0:
        run_starts = invalid_runs[np.where(np.diff(invalid_runs) > 1)[0] + 1]
        run_starts = np.concatenate(([invalid_runs[0]], run_starts))
        run_ends = invalid_runs[np.where(np.diff(invalid_runs) > 1)[0]]
        run_ends = np.concatenate((run_ends, [invalid_runs[-1]]))
        
        # Interpolate gaps shorter than max_samples_gap
        for start, end in zip(run_starts, run_ends):
            gap_length = end - start + 1
            if gap_length <= max_samples_gap:
                # Linear interpolation
                if start > 0 and end < len(az) - 1:
                    az[start:end+1] = np.interp(
                        np.arange(start, end+1),
                        [start-1, end+1],
                        [az[start-1], az[end+1]]
                    )
                    el[start:end+1] = np.interp(
                        np.arange(start, end+1),
                        [start-1, end+1],
                        [el[start-1], el[end+1]]
                    )
    
    # Apply Savitzky-Golay filter with adjusted window length for 500 Hz
    # Window length of 5 (10ms at 500 Hz) and polynomial order of 3
    az_filtered = signal.savgol_filter(az, window_length=51, polyorder=3)
    el_filtered = signal.savgol_filter(el, window_length=51, polyorder=3)
    
    # Calculate quality metrics
    quality_metrics = {
        'percent_signal_loss': np.mean(signal_loss_mask) * 100,
        'percent_high_velocity': np.mean(velocity_mask) * 100,
        'number_of_gaps': len(run_starts) if 'run_starts' in locals() else 0,
        'mean_gap_duration': np.mean(run_ends - run_starts) / sampling_rate * 1000 if 'run_starts' in locals() else 0
    }
    
    # # Plot the original and smoothed data
    # targets_df = location_pix_to_spherical()
    # plt.figure(figsize=(10, 5))
    # plt.plot(az, label='Azimuth')
    # plt.plot(el, label='Elevation')
    # plt.scatter(np.where(signal_loss_mask)[0], az[signal_loss_mask], color='red', label='Signal Loss (Azimuth)', zorder=5)
    # plt.scatter(np.where(signal_loss_mask)[0], el[signal_loss_mask], color='red', label='Signal Loss (Elevation)', zorder=5)
    # plt.legend()
    # plt.xlabel('Time (samples)')
    # plt.ylabel('Degrees')
    # plt.title('Eye Tracking Data with Signal Loss Highlighted')
    # plt.savefig('data_removed.png')
    # plt.clf()

    # plt.figure(figsize=(10, 5))
    # plt.plot(az, el, label='Original', alpha=0.5)
    # plt.plot(az_filtered, el_filtered, label='Smoothed', linestyle='--', alpha=0.5)
    # plt.legend()
    # plt.xlabel('Azimuth')
    # plt.ylabel('Elevation')
    # plt.title('Smoothing with Extreme Value Masking')

    # for _, row in targets_df.iterrows():
    #     circle = plt.Circle(
    #         (row["azimuth"], row["elevation"]),
    #         max(row["azimuth_radius"], row["elevation_radius"]),
    #         color="blue",
    #         fill=False,
    #         linewidth=2
    #     )
    #     plt.gca().add_patch(circle)
    #     #Mark the center of the circle
    #     plt.scatter(row["azimuth"], row["elevation"], color="red", label="Center" if _ == 0 else "")
    # plt.savefig('pre_vs_post_filter.png')
    # plt.clf()
    
    return az_filtered, el_filtered, quality_metrics, location_pix_to_spherical()

def location_pix_to_spherical():
    #Screen resolution in pixels
    resolution = (1920, 1080) 
    #screen size in inches (width, height) 
    screen_size = (24, 13.5)
    #viewing distance in cm
    viewing_distance_cm = 19   

    #convert screen size to cm (1 inch = 2.54 cm)
    screen_size_cm = tuple(dim * 2.54 for dim in screen_size)

    #pixel to cm conversion factors
    width_factor = screen_size_cm[0] / resolution[0]
    height_factor = screen_size_cm[1] / resolution[1]

    #target locations in pixels: x_start, y_start, x_end, y_end
    targets_pixels = [
        [1277, 410, 1453, 670],
        [1075, 761, 1251, 1021],
        [670, 761, 846, 1021],
        [467, 410, 643, 670],
        [670, 59, 846, 319],
        [1075, 59, 1251, 319]
    ]

    targets_df = pd.DataFrame(targets_pixels, columns=["x_start", "y_start", "x_end", "y_end"])

    #compute center of each box in pixels
    targets_df["x_center"] = (targets_df["x_start"] + targets_df["x_end"]) / 2
    targets_df["y_center"] = (targets_df["y_start"] + targets_df["y_end"]) / 2

    #convert pixel centers to cm from screen center
    targets_df["x_cm"] = (targets_df["x_center"] - resolution[0] / 2) * width_factor
    targets_df["y_cm"] = (resolution[1] / 2 - targets_df["y_center"]) * height_factor

    #calculate azimuth and elevation in degrees
    targets_df["azimuth"] = np.degrees(2 * np.arctan(targets_df["x_cm"] / (2 * viewing_distance_cm)))
    targets_df["elevation"] = np.degrees(2 * np.arctan(targets_df["y_cm"] / (2 * viewing_distance_cm)))

    #calculate azimuth and elevation ranges for radii
    targets_df["azimuth_radius"] = abs(
        np.degrees(2 * np.arctan((targets_df["x_start"] - resolution[0] / 2) * width_factor / (2 * viewing_distance_cm))) -
        np.degrees(2 * np.arctan((targets_df["x_end"] - resolution[0] / 2) * width_factor / (2 * viewing_distance_cm)))
    ) / 2

    targets_df["elevation_radius"] = abs(
        np.degrees(2 * np.arctan((resolution[1] / 2 - targets_df["y_start"]) * height_factor / (2 * viewing_distance_cm))) -
        np.degrees(2 * np.arctan((resolution[1] / 2 - targets_df["y_end"]) * height_factor / (2 * viewing_distance_cm)))
    ) / 2

    return targets_df

def read_trials_and_event():
    return pd.read_excel('98_FaceSearchAll_base.xls'), pd.read_csv('FEVENT.csv')

def set_norm(eyelink):
    eyelink['norm'] = np.sqrt(eyelink['azimuth']**2 + eyelink['elevation']**2)
    return eyelink

def plot_saccade_landings(idx,target_center,target_radius,target_df,current_target,saccade_offsets_ts,trial_start,trial_end,filtered_data):
    plt.figure(figsize=(10,10))
    plt.title(f"Trial {idx + 1}: Saccade Endpoints and ROIs")
    plt.xlabel("Azimuth (degrees)")
    plt.ylabel("Elevation (degrees)")

    # Plot the target ROI
    box_corner = (target_center[0] - target_radius, target_center[1] - target_radius)
    target_box = plt.Rectangle(
        box_corner,            #bottom left corner
        2 * target_radius, #width
        2 * target_radius, #height
        color="green",
        fill=False,
        linewidth=2,
        label="Target ROI"
    )
    # target_circle = plt.Circle(
    #     (target_center[0], target_center[1]),
    #     target_radius,
    #     color="green",
    #     fill=False,
    #     linewidth=2,
    #     label="Target ROI"
    # )
    plt.gca().add_patch(target_box)
    plt.scatter(target_center[0], target_center[1], color="green", label="Target Center")

    # Plot all distractors
    for distractor_idx, distractor_row in target_df.iterrows():
        if distractor_idx == current_target - 1:
            continue  # Skip the current target

        distractor_center = distractor_row[['azimuth', 'elevation']].values
        distractor_radius = distractor_row['max_radius']

        box_corner = (distractor_center[0] - distractor_radius, distractor_center[1] - distractor_radius)
        distractor_box = plt.Rectangle(
            box_corner,            #bottom left corner
            2 * distractor_radius, #width
            2 * distractor_radius, #height
            color="blue",
            fill=False,
            linewidth=2,
            label="Distractor ROI" if distractor_idx == 0 else ""
        )
        # distractor_circle = plt.Circle(
        #     (distractor_center[0], distractor_center[1]),
        #     distractor_radius,
        #     color="blue",
        #     fill=False,
        #     linewidth=2,
        #     label="Distractor ROI" if distractor_idx == 0 else ""
        # )
        plt.gca().add_patch(distractor_box)
        plt.scatter(distractor_center[0], distractor_center[1], color="blue", label="Distractor Center" if distractor_idx == 0 else "")

    # Plot saccade endpoints
    label_added = False
    for offset_ts in saccade_offsets_ts:
        if trial_start <= offset_ts <= trial_end:
            # Get the eye position at the saccade offset timestamp
            eye_pos = filtered_data.loc[offset_ts, ['azimuth', 'elevation']].to_numpy()
                # print(offset_ts)
                # print(trial_start)
                # print(trial_end)
                # print(eye_pos)
                # print(eye_pos[-1])
                # print(eye_pos[0])
                # input()
            label = "Saccade End" if not label_added else ""
            label_added = True
            plt.scatter(eye_pos[0], eye_pos[1], color="red", label=label)# if offset_ts == saccade_offsets_ts[0] else "")
    
    plt.legend(loc="best")
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('current_saccade_landings.png',dpi=400)
    plt.clf()