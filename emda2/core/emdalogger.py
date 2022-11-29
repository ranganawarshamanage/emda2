"""
Author: "Rangana Warshamanage, Garib N. Murshudov"
MRC Laboratory of Molecular Biology

This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""

import tabulate
import pandas as pd
import collections.abc


def vec2string(vec):
    return " ".join(("% .3f" % x for x in vec))

def list2string(vec):
    return ", ".join(("%s" % x for x in vec))

def log_newline(fobj):
    print('\n')
    fobj.write('\n')

def log_string(fobj, s):
    print(s)
    fobj.write('%s\n' % s)

def log_vector(fobj, dic):
    for key, val in dic.items():
        if hasattr(val, "__len__"):
            # val is a numpy array
            fobj.write('%s:%s\n' % (key, vec2string(val)))
        elif isinstance(val, collections.abc.Sequence):
            # val is a python list
            fobj.write('%s:%s\n' % (key, vec2string(val)))
        else:
            # val is a scalar
            fobj.write('%s:%s\n' % (key, val))

def log_fsc(fobj, dic):
    keys = list(dic.keys())
    vals = list(dic.values())
    df = pd.DataFrame(dic)
    print(tabulate.tabulate(df, headers=keys, showindex=False))
    resformatter = lambda x: '%4.2f' % x
    fscformatter = lambda x: '%5.3f' % x
    formatters = {}
    for i, key in enumerate(keys):
        if i==0:
            formatters[key] = resformatter
        else:
            formatters[key] = fscformatter
    #formatters = {keys[i]:resformatter if i==0 else fscformatter for i in range(len(keys)) }
    dfAsString = df.to_string(formatters=formatters, header=True, index=False)
    fobj.write(dfAsString)
    fobj.write('\n')


def log_tracebck(fobj, tb):
    fobj.write('\n')
    fobj.write(tb)