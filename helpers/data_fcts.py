import numpy as np
import scipy.signal



"""
CONSTANTS
"""
ETHZ_SENSORS = {
    "RGBD": ["CAM1", "CAM3"],
    "USS": ["USS1", "USS3"],
    "ToF": ["TOF1", "TOF3"],
}
RH2_SENSORS = {
    "RGBD": ["RGBD_1", "RGBD_2", "RGBD_3", "RGBD_4"],
    "USS": ["USS1", "USS2", "USS3", "USS4"],
    "ToF": ["ToF1", "ToF2", "ToF3", "ToF4"],
}



def linInterpolateArray(
    x1:np.array,
    y1:np.array,
    x2:np.array,
    border_condition:str="nan",
):
    """
    Find corresponding y2 values for x2 values by linear interpolation,
    where x1 and y1 are two correlated arrays and used for interpolation.
    Apply original order of x2 to y2.
    Args:
        x1: input x values; np.array of shape (N,)
        y1: input y values; np.array of shape (N,)
        x2: output x values; np.array of shape (M,)
        border_condition: how to handle x2 values that are outside of the range of x1; str
            "nan": return nan for those values
            "nearest": return nearest y1 value for those values
    Returns:
        y2: output y values; np.array of shape (M,)
    """
    x1 = np.copy(x1)
    y1 = np.copy(y1)
    x2 = np.copy(x2)

    # check input
    if x1.shape != y1.shape:
        print(f"ERROR: data_fcts.linInterpolateArray: x1.shape != y1.shape")
    if border_condition not in ["nan", "nearest"]:
        print(f"ERROR: data_fcts.linInterpolateArray: border_condition not in ['nan', 'nearest']")
    
    if np.min(x2)<np.min(x1):
        print(f"Warning: data_fcts.linInterpolateArray: np.min(x2)={np.min(x2)} < np.min(x1)={np.min(x1)}")
        if border_condition == "nan":
            print(f"Warning: data_fcts.linInterpolateArray: returning nan for values outside of x1 range")
        else:
            print(f"Warning: data_fcts.linInterpolateArray: returning nearest y1 value for values outside of x1 range")
    if np.max(x2)>np.max(x1):
        print(f"Warning: data_fcts.linInterpolateArray: np.max(x2)={np.max(x2)} > np.max(x1)={np.max(x1)}")
        if border_condition == "nan":
            print(f"Warning: data_fcts.linInterpolateArray: returning nan for values outside of x1 range")
        else:
            print(f"Warning: data_fcts.linInterpolateArray: returning nearest y1 value for values outside of x1 range")

    # sort x1 and y1 by x1
    idxs_sort1 = np.argsort(x1)
    x1 = x1[idxs_sort1]
    y1 = y1[idxs_sort1]

    # sort x2
    idxs_sort2 = np.argsort(x2)
    x2 = x2[idxs_sort2]

    # find corresponding y2 values for x2 values by linear interpolation
    if border_condition == "nan":
        y2 = np.interp(x2, x1, y1, left=np.nan, right=np.nan)
    else:
        y2 = np.interp(x2, x1, y1, left=y1[0], right=y1[-1])

    # return y1 in original order of x2
    return y2[idxs_sort2]

def linInterpolateNans(
    arr:np.array,
):
    """
    Replace nan values in array by linear interpolation of closest valid values.
    Args:
        arr: input array; np.array of shape (N,)
    Returns:
        arr: array with replaced nan values; np.array of shape (N,)
    """
    arr = np.copy(arr)
    N = arr.shape[0]
    n = np.sum(~np.isnan(arr))

    if n == 0:
        print(f"ERROR: data_fcts.convolveIgnorNan: all values are nan")
        return arr
    
    if n == N:
        return arr

    # find next value above nan values
    arr_val_idxs = np.arange(N)[~np.isnan(arr)] # [0, N-1], (n,)
    cumsum = np.cumsum(~np.isnan(arr)) # (N,)
    next_val_idx_above = arr_val_idxs[np.clip(cumsum, 0, n-1)] # (N,)
    next_val_above = arr[next_val_idx_above] # (N,) 

    arr_val_idxs_inv = np.arange(N)[~np.isnan(np.flip(arr))] # [0, N-1], (n,)
    cumsum_inv = np.cumsum(~np.isnan(np.flip(arr))) # (N,)
    next_val_idx_below = arr_val_idxs_inv[np.clip(cumsum_inv, 0, n-1)] # (N,)
    next_val_idx_below = N - 1 - np.flip(next_val_idx_below)
    next_val_below = arr[next_val_idx_below] # (N,)
      
    # calculate weights for linear interpolation
    next_val_below_dist = (np.arange(N) - next_val_idx_below).astype(np.int64) # (N,)
    next_val_above_dist = (next_val_idx_above - np.arange(N)).astype(np.int64) # (N,)
    next_val_below_dist = np.where(
        next_val_below_dist<=0, 
        np.iinfo(np.int64).max,
        next_val_below_dist,
    )
    next_val_above_dist = np.where(
        next_val_above_dist<=0,
        np.iinfo(np.int64).max,
        next_val_above_dist,
    )
    weigts_below = 1 / next_val_below_dist # (N,)
    weigts_above = 1 / next_val_above_dist # (N,)
    weights_sum = weigts_below + weigts_above # (N,)
    weigts_below = weigts_below / weights_sum # (N,)
    weigts_above = weigts_above / weights_sum # (N,)
    
    # linear interpolation of nan values
    arr_inter = weigts_below * next_val_below + weigts_above * next_val_above # linear interpolation
    arr[np.isnan(arr)] = arr_inter[np.isnan(arr)] # replace nan values by linear interpolation
    return arr

def convolveIgnorNans(
    arr:np.array,
    kernel:np.array,
):
    """
    Convolve array while ignoring nan values e.g. replace nan values by linear interpolation.
    Args:
        arr: input array; np.array of shape (N,)
        kernel: kernel for convolution; np.array of shape (M,)
    Returns:
        arr_conv: convolved array; np.array of shape (N,)
    """
    arr = np.copy(arr)
    kernel = np.copy(kernel)

    # linear interpolate nan values
    arr = linInterpolateNans(arr)

    # convolve array
    return np.convolve(arr, kernel, mode="same")

def smoothIgnoreNans(
    arr:np.ndarray,
    window_size:int,
    polyorder:int=3,
):
    """
    Smooth array by applying a Savitzky-Golay filter. Lineraly interpolate nan values.
    Args:
        arr: input array; np.array of shape (N,)
        window_size: window size for Savitzky-Golay filter; int
        polyorder: polynomial order for Savitzky-Golay filter; int
    Returns:
        arr_smooth: smoothed array; np.array of shape (N,)
    """
    arr = np.copy(arr)

    if arr.shape[0] < window_size:
        return arr

    # linear interpolate nan values
    arr = linInterpolateNans(arr)

    # smooth array
    return scipy.signal.savgol_filter(arr, window_size, polyorder)

def dataConverged(
    arr:np.array,
    threshold:float,
    data_increasing:bool
):
    """
    Verify at which index the data has converged.
    Args:
        arr: input array; np.array of shape (N,)
        threshold: threshold for convergence; float
        data_increasing: whether the data is increasing or decreasing; bool
    Returns:
        idx_converged: index at which the data has converged; int
                        return -1 if data has not converged
    """
    arr = np.copy(arr)

    arr_binary = np.where(
        arr > threshold, 
        1 if data_increasing else 0, 
        0 if data_increasing else 1,
    )
    
    arr_binary = np.cumprod(arr_binary[::-1])[::-1]

    if not np.any(arr_binary):
        return -1 # data has not converged
    return np.argmax(arr_binary)

def sensorName2ID(
    sensor_name:str,
    dataset:str,
):
    """
    Convert sensor name to sensor stack identity.
    Args:
        sensor_name: sensor name or names; str|np.ndarray
        dataset: dataset name; str
    Returns:
        sensor_idx: sensor index or indices; int|np.ndarray
    """
    if dataset == "ETHZ":
        sensors_types = ETHZ_SENSORS
    elif dataset == "RH2":
        sensors_types = RH2_SENSORS
    else:
        print(f"ERROR: data_fcts.sensorName2Idx: dataset = {dataset} not implemented")
        return None

    possible_sensors = []
    for sensors in sensors_types.values():
        possible_sensors += sensors
    
    # convert to tensor
    return_int = False
    if not isinstance(sensor_name, np.ndarray):
        sensor_name = np.array([str(sensor_name)])
        return_int = True

    # check that sensor_name is valid
    check_bool = np.zeros(sensor_name.shape, dtype=np.bool_)
    for s in possible_sensors:
        check_bool = check_bool | (sensor_name == s)
    if not np.all(check_bool):
        print(f"ERROR: data_fcts.sensorName2Idx: sensor_name not in {possible_sensors}")
        return None
    
    # convert sensor_name to sensor_idx
    sensor_idx = np.zeros(sensor_name.shape, dtype=np.uint8)
    for s in possible_sensors:
        sensor_idx[sensor_name == s] = s[-1]

    # convert back to original type
    if return_int:
        sensor_idx = int(sensor_idx[0].item())
    return sensor_idx

def sensorID2Name(
    sensor_id:int,
    sensor_type:str,
    dataset:str,
):
    """
    Convert sensor stack identity to name.
    Args:
        sensor_id: sensor identity; int|np.ndarray
        sensor_type: sensor type; str
        dataset: dataset name; str
    Returns:
        sensor_name: sensor name or names; str|np.ndarray
    """
    if dataset == "ETHZ":
        sensors_types = ETHZ_SENSORS
    elif dataset == "RH2":
        sensors_types = RH2_SENSORS
    else:
        print(f"ERROR: data_fcts.sensorName2Idx: dataset = {dataset} not implemented")
        return None
    
    possible_sensors = sensors_types[sensor_type]

    # convert to tensor
    return_int = False
    if not isinstance(sensor_id, np.ndarray):
        sensor_id = np.array([int(sensor_id)], dtype=np.uint8)
        return_int = True

    # check that sensor_idx is valid
    check_bool = np.zeros(sensor_id.shape, dtype=np.bool_)
    for s in possible_sensors:
        check_bool = check_bool | (sensor_id == int(s[-1]))
    if not np.all(check_bool):
        print(f"ERROR: data_fcts.sensorIdx2Name: sensor_idx not in {possible_sensors}")
        return None
    
    # convert sensor_idx to sensor_name
    sensor_name = np.full(sensor_id.shape, 'None')
    for s in possible_sensors:
        sensor_name[sensor_id == int(s[-1])] = s

    # convert back to original type
    if return_int:
        sensor_name = str(sensor_name[0].item())
    return sensor_name

def downsampleData(
    datas:list,
    num_imgs:int,
    num_imgs_downsampled:int,
):
    """
    Downsample data.
    Args:
        datas: list of data; list of torch tensors [(N*M, ...), ...] or [(N, M, ...),  ...]
        num_imgs: number of images N; int
        num_imgs_downsampled: number of images in downsampled data; int
    Returns:
        datas: list of downsampled data; list of torch tensors [(N_down*M, ...), ...] or [(N_down, M, ...), ...]
    """
    datas = [np.copy(data) for data in datas]
    
    N = num_imgs
    N_down = num_imgs_downsampled
    if N < num_imgs_downsampled:
        print(f"ERROR: data_fcts.downsampleData: N < num_imgs_downsampled")
        return None

    for i, data in enumerate(datas):
        output_shape = data.shape

        # determine number of samples per image
        if data.shape[0] == N:
            M = data.shape[1]
        else:
            M = data.shape[0] // N
            data = data.reshape((N, M, *output_shape[1:]))
        
        # downsample data
        idxs = np.linspace(0, N-1, N_down, dtype=int)
        data = data[idxs]

        # reshape data
        datas[i] = data.reshape((-1, *output_shape[1:]))
    return datas




