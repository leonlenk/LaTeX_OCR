# Credit goes to https://www.kaggle.com/code/kalikichandu/preprossing-inkml-to-png-files for original code.

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize
import xml.etree.ElementTree as ET
import os
import numpy as np
from tqdm import tqdm
import cv2
import collections

def get_traces_data(inkml_file_abs_path):
    

    traces_data = []
    
    tree = ET.parse(inkml_file_abs_path)
    root = tree.getroot()
    doc_namespace = "{http://www.w3.org/2003/InkML}"

#   'Stores traces_all with their corresponding id'
    traces_all = [{'id': trace_tag.get('id'),
    					'coords': [[round(float(axis_coord)) if float(axis_coord).is_integer() else round(float(axis_coord) * 10000) \
    									for axis_coord in coord[1:].split(' ')] if coord.startswith(' ') \
    								else [round(float(axis_coord)) if float(axis_coord).is_integer() else round(float(axis_coord) * 10000) \
    									for axis_coord in coord.split(' ')] \
    							for coord in (trace_tag.text).replace('\n', '').split(',')]} \
    							for trace_tag in root.findall(doc_namespace + 'trace')]

#   'Sort traces_all list by id to make searching for references faster'
    traces_all.sort(key=lambda trace_dict: int(trace_dict['id']))

#   'Always 1st traceGroup is a redundant wrapper'
    traceGroupWrapper = root.find(doc_namespace + 'traceGroup')

    if traceGroupWrapper is not None:
        for traceGroup in traceGroupWrapper.findall(doc_namespace + 'traceGroup'):

            label = traceGroup.find(doc_namespace + 'annotation').text

#    'traces of the current traceGroup'
            traces_curr = []
            for traceView in traceGroup.findall(doc_namespace + 'traceView'):

#     'Id reference to specific trace tag corresponding to currently considered label'
                traceDataRef = int(traceView.get('traceDataRef'))

#     'Each trace is represented by a list of coordinates to connect'
                single_trace = traces_all[traceDataRef]['coords']
                traces_curr.append(single_trace)

            traces_data.append({'label': label, 'trace_group': traces_curr})

    else:
#             'Consider Validation data that has no labels'
        [traces_data.append({'trace_group': [trace['coords']]}) for trace in traces_all]

    return traces_data

def get_gt(inkml_file_abs_path):
    tree = ET.parse(inkml_file_abs_path)
    root = tree.getroot()
    doc_namespace = "{http://www.w3.org/2003/InkML}"
    annotation = root.find(f".//{doc_namespace}annotation[@type='truth']")
    if annotation is not None:
        truth = annotation.text
    else: raise Exception("No truth annotation found.")
    return truth

def inkml2img(input_path, output_path):
#     print(input_path)
#     print(pwd)
    traces = get_traces_data(input_path)
#     print(traces)
    path = input_path.split('/')
    path = path[len(path)-1].split('.')
    path = path[0]+'_'
    file_name = 0
    # Get rid of all matplotlib elements
    plt.axis('off')
    # plt.gca().set_position([0, 0, 1, 1])
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.gca().set_axis_off()
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    
    for elem in traces:
        
#         print(elem)
#         print('-------------------------')
#         print(elem['label'])
        ls = elem['trace_group']
        output_path = output_path  
        
        for subls in ls:
#             print(subls)
            
            data = np.array(subls)
            # raise Exception(data)
            if data.shape[1] > 2:
                data = data[:, :2]
            x,y=zip(*data)
            plt.plot(x,y,linewidth=2,c='black')
    try:
        os.mkdir(output_path)
    except OSError:
#             print ("Folder %s Already Exists" % ind_output_path)
#             print(OSError.strerror)
        pass
    else:
#             print ("Successfully created the directory %s " % ind_output_path)
        pass
#         print(ind_output_path+'/'+path+str(file_name)+'.png')
    input_path_safe = input_path.replace('/', '_') + '_'
    if(os.path.isfile(output_path+'/'+input_path_safe+str(file_name)+'.png')):
        # print('1111')
        file_name += 1
        plt.savefig(output_path+'/'+input_path_safe+str(file_name)+'.png', bbox_inches='tight', dpi=100)
    else:
        plt.savefig(output_path+'/'+input_path_safe+str(file_name)+'.png', bbox_inches='tight', dpi=100)
    plt.gcf().clear()

def ink2img_folder(input_paths, output_path):
    labels = collections.defaultdict(list)
    for input_path in input_paths:
        input_path_safe = input_path.replace('/', '_') + '_'
        files = os.listdir(input_path)
        # ignore all files that don't have the .inkML extension
        files = [file for file in files if file.endswith('.inkml')]
        for file in tqdm(files):
        #     print(file)
            if output_path[-1] != "/": output_path += "/"
            inkML_path = os.path.join(input_path, file)
            try: 
                labels["label"].append(get_gt(inkML_path))
                labels["name"].append((input_path_safe+file).replace('.','_').replace('_inkml', '.inkml')+'_0.png')
                inkml2img(inkML_path, output_path)
            except: print("Error with file: " + str(file) + " in folder: " + str(input_path) + ". Don't worry, this is expected (though there should only be max 2 or 3!).")
    pd.DataFrame(labels).to_csv(output_path + "labels.csv", index=False)