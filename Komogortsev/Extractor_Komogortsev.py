import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def angle_between(v1, v2):
    """
    Computes the angle in degrees between vectors v1 and v2.
    Vectors are assumed to be (N, 3) numpy arrays.
    """
    v1_u = v1 / np.linalg.norm(v1, axis=1)[:, np.newaxis]
    v2_u = v2 / np.linalg.norm(v2, axis=1)[:, np.newaxis]
    
    dot_product = np.sum(v1_u * v2_u, axis=1)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def main_sequence_curve(amp, vmax, c):
    """ Función de Main Sequence: Vpeak = Vmax * (1 - exp(-Amp/c)) """
    return vmax * (1.0 - np.exp(-amp / c))

def extract_cemb_features(df, velocity_threshold=30.0):
    """
    Extracts the optimal 7 CEM-B features (Komogortsev).
    """
    df_valid = df[df['valid_deteccion'] == True].copy()
    if len(df_valid) < 10:
        return None
        
    df_valid.sort_values(by='timestamp_ms', inplace=True)
    
    gaze_vectors = df_valid[['gaze_x', 'gaze_y', 'gaze_z']].values.copy()
    timestamps = df_valid['timestamp_ms'].values
    
    v1 = gaze_vectors[:-1]
    v2 = gaze_vectors[1:]
    
    angles = angle_between(v1, v2)
    
    t1 = timestamps[:-1]
    t2 = timestamps[1:]
    delta_t = (t2 - t1) / 1000.0 # Convert ms to seconds
    
    delta_t[delta_t <= 0] = 0.001
    velocities = angles / delta_t # degrees per second
    is_saccade = velocities > velocity_threshold
    
    fixations = [] 
    saccades = []  
    
    if len(is_saccade) == 0:
        return None
        
    current_state = is_saccade[0]
    duration_accumulator = delta_t[0]
    amplitude_accumulator = angles[0]
    velocity_history = [velocities[0]] if is_saccade[0] else []
    
    for i in range(1, len(is_saccade)):
        if is_saccade[i] == current_state:
            duration_accumulator += delta_t[i]
            if current_state: # Saccade
                amplitude_accumulator += angles[i]
                velocity_history.append(velocities[i])
        else:
            # Change of state
            if current_state: # Saccade ended
                if amplitude_accumulator > 0.0 and len(velocity_history) > 0:
                    mean_vel = np.mean(velocity_history)
                    q_ratio = np.max(velocity_history) / mean_vel if mean_vel > 0 else 0.0
                        
                    saccades.append({
                        'duration': duration_accumulator,
                        'amplitude': amplitude_accumulator,
                        'peak_velocity': np.max(velocity_history),
                        'q_ratio': q_ratio
                    })
            else: # Fixation ended
                if duration_accumulator > 0.0:
                    fixations.append({
                        'duration': duration_accumulator
                    })
            
            # Reset accumulators for new state
            current_state = is_saccade[i]
            duration_accumulator = delta_t[i]
            if current_state:
                amplitude_accumulator = angles[i]
                velocity_history = [velocities[i]]
            else:
                amplitude_accumulator = 0.0
                velocity_history = []
                
    # Finalize last state
    if current_state:
        if amplitude_accumulator > 0.0 and len(velocity_history) > 0:
            mean_vel = np.mean(velocity_history)
            q_ratio = np.max(velocity_history) / mean_vel if mean_vel > 0 else 0.0

            saccades.append({
                'duration': duration_accumulator,
                'amplitude': amplitude_accumulator,
                'peak_velocity': np.max(velocity_history),
                'q_ratio': q_ratio
            })
    else:
        if duration_accumulator > 0.0:
            fixations.append({
                'duration': duration_accumulator
            })
            
    # Calculate the 7 optimal metrics
    fix_count = len(fixations)
    mean_fix_duration = np.mean([f['duration'] for f in fixations]) if fix_count > 0 else 0.0
    
    sac_count = len(saccades)
    mean_sac_duration = np.mean([s['duration'] for s in saccades]) if sac_count > 0 else 0.0
    std_sac_amplitude = np.std([s['amplitude'] for s in saccades]) if sac_count > 1 else 0.0
    std_sac_peak_velocity = np.std([s['peak_velocity'] for s in saccades]) if sac_count > 1 else 0.0
    mean_sac_q_ratio = np.mean([s['q_ratio'] for s in saccades]) if sac_count > 0 else 1.0
    
    ms_vmax = 0.0
    ms_c = 1.0
    
    if sac_count >= 5:
        amplitudes = np.array([s['amplitude'] for s in saccades])
        peak_velocities = np.array([s['peak_velocity'] for s in saccades])
        
        try:
            p0 = [np.max(peak_velocities), np.mean(amplitudes)]
            popt, _ = curve_fit(main_sequence_curve, amplitudes, peak_velocities, p0=p0, bounds=([0, 0.1], [2000, 100]), maxfev=5000)
            ms_vmax, ms_c = popt
        except RuntimeError:
            ms_vmax, ms_c = 0.0, 1.0
        except ValueError:
            ms_vmax, ms_c = 0.0, 1.0
    
    def safe_val(v):
        return 0.0 if pd.isna(v) or np.isnan(v) or np.isinf(v) else v
        
    return {
        'ms_vmax': safe_val(ms_vmax),
        'std_saccade_amplitude': safe_val(std_sac_amplitude),
        'std_saccade_peak_velocity': safe_val(std_sac_peak_velocity),
        'ms_c': safe_val(ms_c),
        'mean_fixation_duration': safe_val(mean_fix_duration),
        'mean_saccade_duration': safe_val(mean_sac_duration),
        'mean_saccade_q_ratio': safe_val(mean_sac_q_ratio)
    }


