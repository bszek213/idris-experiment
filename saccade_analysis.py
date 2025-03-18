from scipy.signal import savgol_filter
import numpy as np
from random import randint
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


"""
1. Peak Detection:
First, you need to have detected saccade peaks using the velocity threshold estimation method.

2. Saccade Onset Detection:
For each detected saccade peak:
   a. Start from the leftmost peak saccade sample.
   b. Search backward in time.
   c. Find the first sample that meets two conditions:
      - Velocity is below the saccade onset threshold: $$v < (\mu_z + 3\sigma_z)$$
      - Velocity is decreasing: $$v_i < v_{i-1}$$
   d. This sample is defined as the saccade onset.

3. Saccade Offset Detection:
For each detected saccade peak:
   a. Start from the rightmost peak saccade sample.
   b. Search forward in time.
   c. Calculate a local noise factor $$\mu_t$$ and $$\sigma_t$$ from a window of samples (40 ms) preceding the current saccade.
   d. Calculate the saccade offset threshold: $$v_{ST_{offset}} = (\mu_z + 3\sigma_z) + \lambda(\mu_t + 3\sigma_t)$$
   e. Find the first sample that meets two conditions:
      - Velocity is below the saccade offset threshold: $$v < v_{ST_{offset}}$$
      - Velocity is decreasing: $$v_i < v_{i-1}$$
   f. This sample is defined as the saccade offset.

4. Additional Checks:
   - Ensure the saccade duration is at least 10 ms (or 12 samples at 1000 Hz sampling rate).
   - Exclude saccades that are preceded by a period where the mean velocity $$\mu_t$$ is greater than the peak threshold $$v_{PT}$$.
"""

def derivatives(df):
    df['vel'] = df['norm'].diff() / (df.index.to_series().diff() / 1000)
    df['accel'] = df['vel'].diff() / (df.index.to_series().diff() / 1000)
    df['el'] = df['vel'].abs()
    df['accel'] = df['accel'].abs()
    return df

def find_velocity_peaks(df):
    pt = randint(100,300)
    for n in range(len(df)):
        sub_df = df[df['vel'] < pt]
        mu = sub_df['vel'].mean()
        sig = sub_df['vel'].std()
        pt_prev = pt
        pt = mu + (6*sig)
        if abs(pt - pt_prev) < 1:
            return pt

def detect_saccades(df,pt):
    #store saccade onsets and offsets
    saccade_onsets = []
    saccade_offsets = []
    saccade_onsets_ts = []
    saccade_offsets_ts = []
    i = 0
    while i < len(df):

      #detectg peak where velocity above threshold
      if df['vel'].iloc[i] >= pt:
         saccade_peak = i
         sacc_init, sacc_end = False, False

         #go backward to find saccade onset
         for j in range(saccade_peak, 0, -1):
               if j + 1 < len(df):
                  if (df['vel'].iloc[j] < (df['vel'].mean() + 3*df['vel'].std()) 
                     and (df['vel'].iloc[j] - df['vel'].iloc[j + 1]) >= 0):
                     saccade_onsets.append(j)
                     sacc_init = True
                     break

         # plt.plot(df.index[max(0, saccade_peak - 100): min(len(df), saccade_peak + 100)],
         #          df['az_vel_filter'].iloc[max(0, saccade_peak - 100): min(len(df), saccade_peak + 100)])
         # plt.scatter(df.index[saccade_onsets[0]],df['az_vel_filter'].iloc[saccade_onsets[0]],marker='*',c='red')
         # plt.show()

         #go forward to find saccade offset
         #need to check the last 40 ms
         time_to_samples = int(40 / int(np.nanmean(df.index.to_series().diff())))
         for j in range(saccade_peak, len(df) - 1) :
               #if saccade init is found
               if sacc_init:
                  mu_t = df['vel'].iloc[max(0, j-time_to_samples):j].mean()
                  sig_t = df['vel'].iloc[max(0, j-time_to_samples):j].std()

                  v_st_offset = (0.1*(df['vel'].mean() + 3*df['vel'].std())) + (0.5*(mu_t + 3*sig_t))

               #   plt.plot(df.index[max(0, saccade_peak - 100): min(len(df), saccade_peak + 100)],
               #            df['az_vel_filter'].iloc[max(0, saccade_peak - 100): min(len(df), saccade_peak + 100)])
               #   plt.hlines(v_st_offset,xmin=df.index[max(0, saccade_peak - 100)]
               #              ,xmax=df.index[min(len(df), saccade_peak + 100)])
               #   plt.scatter(df.index[saccade_onsets[0]],df['az_vel_filter'].iloc[saccade_onsets[0]],marker='*',c='red')
               #   plt.show()
                  
                  if (df['vel'].iloc[j] < v_st_offset 
                     and (df['vel'].iloc[j] - df['vel'].iloc[j + 1]) <= 0):
                     saccade_offsets.append(j)
                     sacc_end = True
                     break

         # #check saccade duration
         if sacc_init and sacc_end:
            duration = df.index[saccade_offsets[-1]] - df.index[saccade_onsets[-1]]
            if duration >= 12:
               saccade_offsets_ts.append(df.index[saccade_offsets[-1]])
               saccade_onsets_ts.append(df.index[saccade_onsets[-1]])
               # plt.figure(figsize=(15,15))
               # plt.plot(df.index[max(0, saccade_peak - 100): min(len(df), saccade_peak + 100)],
               # df['az_vel_filter'].iloc[max(0, saccade_peak - 100): min(len(df), saccade_peak + 100)])
               # plt.title(f"Saccade Amplitude {df['az'].iloc[max(0, saccade_peak - 100): min(len(df), saccade_peak + 100)].abs().mean()} degrees")
               # plt.scatter(df.index[saccade_onsets[-1]],df['az_vel_filter'].iloc[saccade_onsets[-1]],s=150,marker='*',c='red')
               # plt.scatter(df.index[saccade_offsets[-1]],df['az_vel_filter'].iloc[saccade_offsets[-1]],s=150 ,marker='*',c='green')
               # plt.show()

               #move to end of current saccade to avoid overlap
               i = saccade_offsets[-1]
            else:
               if saccade_onsets:
                  #remove onset if invalid
                  saccade_onsets.pop()  
               if saccade_offsets:
                  #remove offset if invalid
                  saccade_offsets.pop()  
      i += 1

    return saccade_onsets_ts, saccade_offsets_ts

def detect_saccades_eyelink(df):
    sacc_df = df[(df['codestring'] == "STARTSACC") | (df['codestring'] == "ENDSACC")]
    sacc_df = sacc_df[sacc_df['codestring'] == "ENDSACC"]
    return sacc_df[['sttime','entime','codestring']]
   
