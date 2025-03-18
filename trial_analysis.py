
def extract_time_stamps_for_each_trial(event_df):
    search_rows = event_df[event_df['message'].str.contains(r'SearchStart \d+', na=False, regex=True)]
    result = search_rows[['message', 'sttime']]
    result['sttime'] = result['sttime'].astype(int)
    return result.reset_index(drop=True)