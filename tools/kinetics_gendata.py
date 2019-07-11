
import os
import sys
import pickle
import argparse

import numpy as np
from numpy.lib.format import open_memmap

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from feeder.feeder_kinetics import Feeder_kinetics

toolbar_width = 30

def print_toolbar(rate, annotation=''):
    # setup toolbar
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')


def end_toolbar():
    sys.stdout.write("\n")


def gendata(
        data_path,
        label_path,
        data_out_path,
        label_out_path,
        num_person_in=5,  #observe the first 5 persons 
        num_person_out=2,  #then choose 2 persons with the highest score 
        max_frame=300):

    feeder = Feeder_kinetics(
        data_path=data_path,
        label_path=label_path,
        num_person_in=num_person_in,
        num_person_out=num_person_out,
        window_size=max_frame)

    sample_name = feeder.sample_name
    sample_label = []

    '''fp = open_memmap(
        data_out_path,
        dtype='float32',
        mode='w+',
        shape=(len(sample_name), 3, max_frame, 18, num_person_out))'''
    skeleton_id_1_origin = '408092820'
    skeleton_id_2_origin = '975158737'

    for i, s in enumerate(sample_name):
        data, label = feeder[i]
        print_toolbar(i * 1.0 / len(sample_name),
                      '({:>5}/{:<5}) Processing data: '.format(
                          i + 1, len(sample_name)))
        #fp[i, :, 0:data.shape[1], :, :] = data
        if label < 10:
            action_id = '00' + str(label)
        elif label >=10 and label < 100:
            action_id = '0' + str(label)
        else:
            action_id = str(label)
        skeleton_id_1  = skeleton_id_1_origin + str(i)
        skeleton_id_2  = skeleton_id_2_origin + str(i)
        file_name = data_out_path + '/' + action_id + '_' + s.split('.')[0] + '.skeleton'
        f = open(file_name, "w+")
        f.write(str(300) + '\n')
        for j in range(300):
            '''f.write(str(2) + '\n')
            f.write(skeleton_id_1 + '\n')
            f.write(str(18) + '\n')
            for k in range(18):
                output_data_1 = str(data[0][j][k][0]) + " " + str(data[1][j][k][0]) + " " + str(data[2][j][k][0]) + ' 0 0 0 0 0 0 0 0'
                f.write(str(output_data_1) + '\n')
            f.write(skeleton_id_2 + '\n')
            f.write(str(18) + '\n')
            for l in range(18):
                output_data_2 = str(data[0][j][l][1]) + " " + str(data[1][j][l][1]) + " " + str(data[2][j][l][1]) + ' 0 0 0 0 0 0 0 0'
                f.write(str(output_data_2) + '\n')'''
            f.write(str(2) + '\n')
            f.write(skeleton_id_1 + '\n')
            f.write(str(25) + '\n')
            for k in range(2):
                body_1_joint_1 = str((data[0][j][8][k] + data[0][j][11][k])/2) + " " + str((data[1][j][8][k] + data[1][j][11][k])/2) + " " + str((data[2][j][8][k] + data[2][j][11][k])/2) + ' 0 0 0 0 0 0 0 0'
                body_1_joint_2 = str(((data[0][j][8][k] + data[0][j][11][k])/2 + data[0][j][1][k])/2) + " " + str(((data[1][j][8][k] + data[1][j][11][k])/2 + data[1][j][1][k])/2) + " " + str(((data[2][j][8][k] + data[2][j][11][k])/2 + data[2][j][1][k])/2) + ' 0 0 0 0 0 0 0 0'
                body_1_joint_3 = str((data[0][j][0][k] + data[0][j][1][k])/2) + " " + str((data[1][j][0][k] + data[1][j][1][k])/2) + " " + str((data[2][j][0][k] + data[2][j][1][k])/2) + ' 0 0 0 0 0 0 0 0'
                body_1_joint_4 = str(data[0][j][0][k]) + " " + str(data[1][j][0][k]) + " " + str(data[2][j][0][k]) + ' 0 0 0 0 0 0 0 0'
                body_1_joint_5 = str(data[0][j][5][k]) + " " + str(data[1][j][5][k]) + " " + str(data[2][j][5][k]) + ' 0 0 0 0 0 0 0 0'
                body_1_joint_6 = str(data[0][j][6][k]) + " " + str(data[1][j][6][k]) + " " + str(data[2][j][6][k]) + ' 0 0 0 0 0 0 0 0'
                body_1_joint_7 = str(data[0][j][7][k]) + " " + str(data[1][j][7][k]) + " " + str(data[2][j][7][k]) + ' 0 0 0 0 0 0 0 0'
                body_1_joint_8 = str(data[0][j][7][k]) + " " + str(data[1][j][7][k]) + " " + str(data[2][j][7][k]) + ' 0 0 0 0 0 0 0 0'
                body_1_joint_9 = str(data[0][j][2][k]) + " " + str(data[1][j][2][k]) + " " + str(data[2][j][2][k]) + ' 0 0 0 0 0 0 0 0'
                body_1_joint_10 = str(data[0][j][3][k]) + " " + str(data[1][j][3][k]) + " " + str(data[2][j][3][k]) + ' 0 0 0 0 0 0 0 0'
                body_1_joint_11 = str(data[0][j][4][k]) + " " + str(data[1][j][4][k]) + " " + str(data[2][j][4][k]) + ' 0 0 0 0 0 0 0 0'
                body_1_joint_12 = str(data[0][j][4][k]) + " " + str(data[1][j][4][k]) + " " + str(data[2][j][4][k]) + ' 0 0 0 0 0 0 0 0'
                body_1_joint_13 = str(data[0][j][11][k]) + " " + str(data[1][j][11][k]) + " " + str(data[2][j][11][k]) + ' 0 0 0 0 0 0 0 0'
                body_1_joint_14 = str(data[0][j][12][k]) + " " + str(data[1][j][12][k]) + " " + str(data[2][j][12][k]) + ' 0 0 0 0 0 0 0 0'
                body_1_joint_15 = str(data[0][j][13][k]) + " " + str(data[1][j][13][k]) + " " + str(data[2][j][13][k]) + ' 0 0 0 0 0 0 0 0'
                body_1_joint_16 = str(data[0][j][13][k]) + " " + str(data[1][j][13][k]) + " " + str(data[2][j][13][k]) + ' 0 0 0 0 0 0 0 0'
                body_1_joint_17 = str(data[0][j][8][k]) + " " + str(data[1][j][8][k]) + " " + str(data[2][j][8][k]) + ' 0 0 0 0 0 0 0 0'
                body_1_joint_18 = str(data[0][j][9][k]) + " " + str(data[1][j][9][k]) + " " + str(data[2][j][9][k]) + ' 0 0 0 0 0 0 0 0'
                body_1_joint_19 = str(data[0][j][10][k]) + " " + str(data[1][j][10][k]) + " " + str(data[2][j][10][k]) + ' 0 0 0 0 0 0 0 0'
                body_1_joint_20 = str(data[0][j][10][k]) + " " + str(data[1][j][10][k]) + " " + str(data[2][j][10][k]) + ' 0 0 0 0 0 0 0 0'
                body_1_joint_21 = str(data[0][j][1][k]) + " " + str(data[1][j][1][k]) + " " + str(data[2][j][1][k]) + ' 0 0 0 0 0 0 0 0'
                body_1_joint_22 = str(data[0][j][7][k]) + " " + str(data[1][j][7][k]) + " " + str(data[2][j][7][k]) + ' 0 0 0 0 0 0 0 0'
                body_1_joint_23 = str(data[0][j][7][k]) + " " + str(data[1][j][7][k]) + " " + str(data[2][j][7][k]) + ' 0 0 0 0 0 0 0 0'
                body_1_joint_24 = str(data[0][j][4][k]) + " " + str(data[1][j][4][k]) + " " + str(data[2][j][4][k]) + ' 0 0 0 0 0 0 0 0'
                body_1_joint_25 = str(data[0][j][4][k]) + " " + str(data[1][j][4][k]) + " " + str(data[2][j][4][k]) + ' 0 0 0 0 0 0 0 0'
                f.write(body_1_joint_1 + '\n')
                f.write(body_1_joint_2 + '\n')
                f.write(body_1_joint_3 + '\n')
                f.write(body_1_joint_4 + '\n')
                f.write(body_1_joint_5 + '\n')
                f.write(body_1_joint_6 + '\n')
                f.write(body_1_joint_7 + '\n')
                f.write(body_1_joint_8 + '\n')
                f.write(body_1_joint_9 + '\n')
                f.write(body_1_joint_10 + '\n')
                f.write(body_1_joint_11 + '\n')
                f.write(body_1_joint_12 + '\n')
                f.write(body_1_joint_13 + '\n')
                f.write(body_1_joint_14 + '\n')
                f.write(body_1_joint_15 + '\n')
                f.write(body_1_joint_16 + '\n')
                f.write(body_1_joint_17 + '\n')
                f.write(body_1_joint_18 + '\n')
                f.write(body_1_joint_19 + '\n')
                f.write(body_1_joint_20 + '\n')
                f.write(body_1_joint_21 + '\n')
                f.write(body_1_joint_22 + '\n')
                f.write(body_1_joint_23 + '\n')
                f.write(body_1_joint_24 + '\n')
                f.write(body_1_joint_25 + '\n')
                if k != 1:
                    f.write(skeleton_id_2 + '\n')
                    f.write(str(25) + '\n')
        f.close()
        #sample_label.append(label)

    '''with open(label_out_path, 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Kinetics-skeleton Data Converter.')
    parser.add_argument(
        '--data_path', default='data/Kinetics/kinetics-skeleton')
    parser.add_argument(
        '--out_folder', default='data/Kinetics/kinetics-skeleton')
    arg = parser.parse_args()

    #part = ['train', 'val']
    part = ['val']
    for p in part:
        data_path = '{}/kinetics_{}'.format(arg.data_path, p)
        label_path = '{}/kinetics_{}_label.json'.format(arg.data_path, p)
        #data_out_path = '{}/{}_data.npy'.format(arg.out_folder, p)
        data_out_path = arg.out_folder
        label_out_path = '{}/{}_label.pkl'.format(arg.out_folder, p)

        if not os.path.exists(arg.out_folder):
            os.makedirs(arg.out_folder)
        gendata(data_path, label_path, data_out_path, label_out_path)
