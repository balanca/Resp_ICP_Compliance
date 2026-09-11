global_key = 'all'


detect_resp_params = {
    'N_cycles_sliding_sd':30, # number of resp cycles in the sliding window computing sd of cycle frequencies
    'threshold_controlled_ventilation_sd_cpm':'auto', # threshold in cycles per minute of SD of sliding cycle freqs to say if ventilation is controlled (True) or assisted (False)
    'N_cycles_sliding_ventilation_bool':100, # number of resp cycles in the sliding window computing averaging boolean classifying of cycles (controlled or not)
}

detect_ecg_params = {}

detect_icp_params = {
    'lowcut':0.1,
    'highcut':10,
    'order':4,
    'ftype':'butter',
    'peak_prominence' : 0.5,
    'h_distance_s' : 0.3,
    'rise_amplitude_limits' : (0,20),
    'amplitude_at_trough_low_limit' : -10,
}

ratio_P1P2_params = {
    'down_sample':False,
    'win_compute_duration_hours':1
}

heart_resp_in_icp_params = {
    'spectrogram_win_size_secs':60,
    'resp_fband':(0.12,0.6),
    'heart_fband':(0.8,2.5),
    'rolling_N_time_spectrogram':5,
}

icp_by_resp_cycle_params = {
    'detect_resp_params':detect_resp_params,
    'highcut':0.5, 
    'order':4, 
    'ftype':'butter',
    'segmentation_deformation':'bi', # mono or bi segment
    'points_per_cycle':100
}

icp_pulse_by_resp_cycle_params = {
    'detect_resp_params':detect_resp_params,
    'detect_icp_params':detect_icp_params, 
    'segmentation_deformation':'bi', # mono or bi segment
    'points_per_cycle':100
}
