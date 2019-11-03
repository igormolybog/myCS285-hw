import os
folders = [folder for folder in os.listdir('.') if os.path.isdir(folder) and folder.split('_')[0] == 'mb']

import re
get_name = lambda f: "".join(re.split('_|\.', f)[3])

import tensorflow as tf

def parse_tf_events_file(filename):
    events = {'Eval_AverageReturn' : [],
              'Eval_StdReturn' : [],
              'Train_AverageReturn' : [],
              'Train_StdReturn' : []}
    keys = events.keys()
    for e in tf.train.summary_iterator(filename):
        for v in e.summary.value:
            if v.tag in keys:
                events[v.tag].append(v.simple_value)
    events['Steps'] = list(range(len(events[list(keys)[0]])))
    return events

keys = ['ensemble', 'horizon', 'numseq']
x1 = {key : {get_name(folder) : parse_tf_events_file(folder+'/'+next(filter(lambda x: x.split('.')[-1]=='X1', os.listdir(folder)))) for folder in folders if get_name(folder).startswith(key)} for key in keys}

import matplotlib.pyplot as plt
import numpy as np
def plot(events, name):
    plt.plot(events['Steps'], events['Eval_AverageReturn'], label=name)
    plt.fill_between(events['Steps'],
                 np.subtract(events['Eval_AverageReturn'], events['Eval_StdReturn']),
                 np.add(events['Eval_AverageReturn'], events['Eval_StdReturn']), alpha=0.1)

for exp_name, exp in x1.items():
    for name, events in exp.items():
        print(name)
        plot(events, name)

    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Average Return")
    plt.title(exp_name)
    plt.savefig(exp_name, dpi=600)
    plt.clf()    
