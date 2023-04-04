import json
import pickle
from typing import Dict, List, Union
import numpy as np


def read_txt(
        fname: str
    ) -> str:
    with open(fname, "r") as fn:
        fcont = fn.read()
    return fcont


def read_binary(
        fname: str
    ) -> str:
    with open(fname, "rb") as fn:
        fcont = fn.read()
    return fcont
    

def write_txt(
        fname: str, 
        fcont: str
    ):
    with open(fname, "w") as fn:
        fn.write(fcont)


def write_binary(
        fname: str, 
        fcont: str
    ):
    with open(fname, "wb") as fn:
        fn.write(fcont)


def load_txt(
        fname: str,
        dtype = float,
        comments: List[str] = ["#"]
    ):
    data = np.loadtxt(fname, comments=comments, dtype=dtype)
    return data

def save_txt(
        fname: str,
        fcont: Union[np.ndarray, List],
        fmt: str = "%.6e"
    ):
    np.savetxt(fname, fcont, fmt=fmt)


def load_json(
        fname: str
    ) -> Dict:
    with open(fname, "r") as fn:
        jdata = json.load(fn)
    return jdata


def dump_json(
        fname: str, 
        fcont: Dict,
        indent: int = 4
    ):
    with open(fname, "w") as fn:
        json.dump(fcont, fn, indent=indent)


def load_pkl(
        fname: str
    ):
    with open(fname, "rb") as ff:
        data = pickle.load(ff) 
    return data


def save_pkl(
        fname,
        obj
    ):
    with open(fname, "wb") as ff:
        pickle.dump(obj, fname)
        
def loaddata(dataset):

    data = np.load(dataset,allow_pickle = True)
    X1 = data['positions']
    cell = data['box']
    topology = data["topology"]
    assert (cell[0][0][0,0] == cell[0][0][1,1])
    L = cell[0][0][0,0]
    traj_length, n, dim = X1.shape[1], X1.shape[2], X1.shape[3]
    X1 = X1.reshape(X1.shape[0], X1.shape[1],n*dim)

    print (X1.shape, L)
    print (np.min(X1), np.max(X1))
    X1 -= L * np.floor(X1/L)
    
    return X1, traj_length, n, dim, L, topology