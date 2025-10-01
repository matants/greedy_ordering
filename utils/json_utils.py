import numpy as np


def array_to_jsonable(a):
    a = np.asarray(a)
    return {"dtype": str(a.dtype), "shape": a.shape, "data": a.ravel().tolist()}


def array_from_jsonable(obj):
    a = np.array(obj["data"], dtype=np.dtype(obj["dtype"]))
    return a.reshape(obj["shape"])
