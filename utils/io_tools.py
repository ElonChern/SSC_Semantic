import torch



def dict_to(_dict, device, dtype):
  '''

  '''
  for key, value in _dict.items():
    if type(_dict[key]) is dict:
      _dict[key] = dict_to(_dict[key], device, dtype)
    else:
      if type(_dict[key]) is torch.Tensor:
        _dict[key] = _dict[key].to(device=device, dtype=dtype, non_blocking=True)
      else:
        pass     
  return _dict