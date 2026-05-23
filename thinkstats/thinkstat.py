

def underride(d, **option):
    """
    Add key value pairs to d only if key is not in d
    
    Args:
        d: dict to add options to
        **option: keyword args to add to d

    Returns:
        dict: Updated dictionary with new key-value pairs.
    """
    for key, val in option.items():
        d.setdefault(key, val)


    return d