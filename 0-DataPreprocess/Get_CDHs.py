# generate CDHs.

import numpy as np
import sys
import os
from plyfile import PlyData,PlyElement
import pandas as pd
import csv 
import json
from rdkit import Chem
from rdkit.Chem import Descriptors
import math
from multiprocessing import Pool
import matplotlib.pyplot as plt
import glob

import warnings
warnings.filterwarnings("ignore")

def process_ply_file(file_path):
    '''
    Process crystal isosurfaces
    '''
    plydata = PlyData.read(file_path)
    di = plydata['vertex']['d_i']
    de = plydata['vertex']['d_e'] 
    d_norm_i = plydata['vertex']['d_norm_i']
    d_norm_e = plydata['vertex']['d_norm_e']
    # element_i = plydata['vertex']['element_a']
    # element_e = plydata['vertex']['element_b']

    return di,de,d_norm_i,d_norm_e

def process_pro_ply_file(file_path):
    '''
    Process molecular isosurfaces
    '''
    plydata = PlyData.read(file_path)
    di = plydata['vertex']['d_i']
    d_norm_i = plydata['vertex']['d_norm_i']
    # d_norm_pi = plydata['vertex']['d_norm_pi'] 
    # element_i = plydata['vertex']['element_a']
    # idx_a = plydata['vertex']['idx_a']

    return di,d_norm_i

def process_single_data(item):
    if item.split(',')[0] != 'id':
        try:
            id = item.split(',')[0][0:7]
            base_names = glob.glob('HS_0530/id_' + id + '_*.ply')
            base_names2 = glob.glob('HS_pro_0530/id_' + id + '_*.ply')
            if len(base_names) == 0:
                return None

            all_di, all_de, all_element_i, all_element_e, all_idx_a, all_idx_b, all_d_norm_i, all_d_norm_e = [], [], [], [], [], [], [], []
            all_pro_di, all_pro_element_i, all_pro_idx_a, all_pro_d_norm_i, all_pro_d_norm_pi = [], [], [], [], []
            for base_name in base_names:
                di, de, d_norm_i, d_norm_e = process_ply_file(base_name)
                all_di.extend(di)
                all_de.extend(de)
                all_d_norm_i.extend(d_norm_i)
                all_d_norm_e.extend(d_norm_e)

            for base_name2 in base_names2:
                pro_di, pro_d_norm_i = process_pro_ply_file(base_name2)
                all_pro_di.extend(pro_di)
                all_pro_d_norm_i.extend(pro_d_norm_i)

            bins_range_di = np.linspace(0.0, 3.2, 65)
            bins_range_din = np.linspace(-2, 2, 65)
            bins_range_din2 = np.linspace(-1, 2, 65)

            hist_0, _, _ = np.histogram2d(all_di, all_de, bins=[bins_range_di, bins_range_di])
            hist_3, _, _ = np.histogram2d(all_d_norm_i, all_d_norm_e, bins=[bins_range_din, bins_range_din])
            hist_5, _, _ = np.histogram2d(all_pro_di, all_pro_d_norm_i, bins=[bins_range_di, bins_range_din2])
            target = item.split(',')[1][:-1]
            return (hist_0/len(base_names), hist_3/len(base_names),hist_5/len(base_names2),target,id)
             
        except Exception as e:
            print(f"Error processing {id}: {str(e)}")
    return None

def main():
    with open('targets_Eg.csv', 'r') as data_csv:
        HS_data = data_csv.readlines()
    print("Dataset number:", len(HS_data))

    with Pool(processes=64) as pool:  
        results = pool.map(process_single_data, HS_data[1:])  
    filtered_results = [result for result in results if result is not None]
    CDH2 = np.array([result[0] for result in filtered_results])
    CDH3 = np.array([result[1] for result in filtered_results])
    CDH1 = np.array([result[2] for result in filtered_results])
    targets_band = [result[3] for result in filtered_results]

    np.save('CDH2.npy', CDH2)
    np.save('target.npy', targets_band)
    np.save('CDH3.npy', CDH3)
    np.save('CDH1.npy', CDH1)

if __name__ == '__main__':
    main()
    
