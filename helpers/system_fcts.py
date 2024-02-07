import sys
import torch
import numpy as np

if torch.cuda.is_available():
    import nvidia_smi

def checkGPUMemory(
    memory_threshold:int=2e9,
    print_memory:bool=True,
    device_index:int=0,
):
    """
    Check if there is enough memory on the GPU for one more training cycle.
    Args:
        memory_threshold: threshold in bytes; int
        print_memory: print memory usage; bool
        device_index: index of gpu; int
    Returns:
        not_enough_memory: bool
    """
    if not torch.cuda.is_available():
        return False
    
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device_index) # gpu id 0
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

    if print_memory:
        print(f"Free memory: {(info.free/1e6):.2f}Mb / {(info.total/1e6):.2f}Mb = {(info.free/info.total):.3f}%")

    not_enough_memory = info.free < memory_threshold
    if not_enough_memory and print_memory:
        print("EXIT: Used memory is too high.\n\n")

    nvidia_smi.nvmlShutdown()
    return not_enough_memory

def get_size(
    obj, 
    seen=None
):
    """
    Recursively finds size of objects.
    Source: https://goshippo.com/blog/measure-real-size-any-python-object/
    Args:
        obj: object to find size of
        seen: helper object to keep track of seen objects
    Returns:
        size of object in bytes
    """
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def moveToRecursively(
    obj,
    destination:str,
    depth:int=0,
    max_depth:int=6,
):
    """
    Move object recursively to cpu, delete it and free its memory.
    Args:
        obj: object to free its memory
        destination: destination device; str
        depth: current depth of recursion; int
        max_depth: maximum depth of recursion; int
    """
    if depth > max_depth:
        return
    depth += 1

    if hasattr(obj, "to"):
        obj.to(destination)
        return
    
    if isinstance(obj, np.ndarray):
        return
    
    if hasattr(obj, "__dict__"):
        for sub_obj in obj.__dict__.values():
            moveToRecursively(
                obj=sub_obj,
                destination=destination,
                depth=depth,
                max_depth=max_depth,
            )
    
    if isinstance(obj, dict):
        for sub_obj in obj.values():
            moveToRecursively(
                obj=sub_obj,
                destination=destination,
                depth=depth,
                max_depth=max_depth,
            )

    if isinstance(obj, list) or isinstance(obj, tuple):
        for sub_obj in obj:
            moveToRecursively(
                obj=sub_obj,
                destination=destination,
                depth=depth,
                max_depth=max_depth,
            )
