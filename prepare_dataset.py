#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Creation date: July 08 2022
@author: Jiawei Li, Technische Universität Dresden
"""

import fnmatch
import logging
import numpy as np
import os
import pandas as pd
import shutil
import sys
import scipy.io


def cfar_rect(rd_map, alpha_in=None):
    '''
    2 dimentional cfar
    range boundary expansion: mean value
    local maximum: guard region
    Doppler boundary expansion: other side value due to circularity.
    '''
    rg_size, dp_size = rd_map.shape
    rd_map_bottom = np.min(rd_map)
    rd_map_mean = np.mean(rd_map)
    target_list = []
    far = 0.1  # false alarm rate
    # far       #alpha
    # 0.1      2.59
    # 0.01     5.85(P,axis=0)(V,axis=1)(C,axis=1)
    # 0.001    9.95
    # 0.0001  15.12
    # 0.00001 21.62
    num_train = 10  # both sides total
    num_half_train = num_train // 2
    num_guard = 2  # both sides total
    num_half_guard = num_guard // 2
    num_side = num_half_train + num_half_guard
    train_cells = (num_train + num_guard + 1) ** 2 - (num_guard + 1) ** 2

    # create expanded rdmap
    rdmap_ex = np.zeros((rg_size + num_train + num_guard, dp_size + num_train + num_guard))
    rdmap_ex[:num_side, :] = rd_map_mean
    rdmap_ex[-num_side:, :] = rd_map_mean
    rdmap_ex[num_side:-num_side, :num_side] = rd_map[:, -num_side:]
    rdmap_ex[num_side:-num_side, -num_side:] = rd_map[:, :num_side]
    rdmap_ex[num_side:num_side + rg_size, num_side:num_side + dp_size] = rd_map

    # alpha can be from 'far' or directly given by alpha_in
    if alpha_in is None:
        alpha = num_train * (far ** (-1 / num_train) - 1)
    else:
        alpha = alpha_in

    for i in range(num_side, rg_size + num_side):  # Explore range not from zero
        for j in range(num_side, dp_size + num_side):  # Explore Doppler not from zero

            if rdmap_ex[i, j] == np.max(rdmap_ex[i - num_guard:i + num_guard + 1,
                                        j - num_guard:j + num_guard + 1]):  # only compare threshold with the local maximum
                sum1 = np.sum(rdmap_ex[i - num_side:i + num_side + 1, j - num_side:j + num_side + 1])
                sum2 = np.sum(
                    rdmap_ex[i - num_half_guard:i + num_half_guard + 1, j - num_half_guard:j + num_half_guard + 1])
                p_noise = (sum1 - sum2) / train_cells  # The guard cells are discarded

                threshold = alpha * p_noise
                if rdmap_ex[i, j] > threshold:
                    target_list.append([i - num_side, j - num_side, rdmap_ex[i, j]])

    return target_list  # each item is (rg_idx, dp_idx, magnitude)


class PrepareDataset:
    """
    prepare datasets for KCNN. Convent original data Xcube.mat, Xbf.mat and label.mat to .npy and label.csv.

    minimum, maximum: minimal and maximal velocity which is matched with factors.
    threshold: maximal detected velocity of radar (Vmax).
    labels_name: should be match with label.csv. labels_name decides if this dataset for 1 or multiple targets!
    normalize: if do normalization
    scaling_unit_length: if do scaling on unit length
    quan_norm: if do 16bit normalization
    test_data: if this dataset testdataset. if it is test_data, the abs will not be calculated
    include_Xcube: if this dataset has Xcube.mat. if it has Xcube.mat, Xcube.mat will be operated
    """

    def __init__(self, minimum: int = -5, maximum: int = 3, threshold=18.24, labels_name=None, normalize=False,
                 scaling_unit_length=False, quan_norm=False, test_data=False,
                 include_Xcube=False):

        if labels_name is None:
            self.labels_name = ['record_name', 'frame_name', 'num_target', 't1_true_range',
                                't1_det_range', 't1_true_velocity', 't1_det_velocity', 't1_factor', 't1_classID']
        else:
            self.labels_name = labels_name

        self.minimum = minimum
        self.maximum = maximum
        self.threshold = threshold
        self.normalize = normalize
        self.scaling_unit_length = scaling_unit_length
        self.quan_norm = quan_norm
        self.test_data = test_data
        self.include_Xcube = include_Xcube

    def _load_mat(self, record_path, mat_name):
        """
        read one mat data
        """
        return scipy.io.loadmat(record_path + '/' + mat_name)

    def _load_all_mats(self, record_path):
        """
        read all mat data
        """
        label = self._load_mat(record_path, 'label.mat')
        Xbf = self._load_mat(record_path, 'Xbf.mat')
        if self.include_Xcube:
            Xcubes = self._load_mat(record_path, 'Xcubes.mat')
            return label, Xbf, Xcubes
        else:
            return label, Xbf

    def _creat_dir(self, file_path: str):
        """
        create directory for saving data
        """
        if os.path.exists(file_path):
            logging.info(f'{file_path} exists.')
        else:
            # os.mkdir(file_path)
            os.makedirs(file_path, exist_ok=True)
            logging.info(f'{file_path} created.')

    def _creat_dict(self):
        """
        crate labels list matched with label.mat
        :return: labels list
        """
        labels = {}
        for i in range(len(self.labels_name)):
            labels[self.labels_name[i]] = []

        return labels

    def _velocity_range(self):
        '''
        determine which velocity range the velocity should be into
        :return: velocity range
        '''
        if self.minimum % 2 == 0 or self.maximum % 2 == 0 or self.minimum == self.maximum:
            print('Minimum und maxinum should be odd number and should not be same!')
            sys.exit()
        else:
            if self.minimum > self.maximum:
                logging.info('Given min > max')
                v_range = [[self.maximum + 2 * x, self.maximum + 2 * (x + 1)] for x in
                           range(int((self.minimum - self.maximum) / 2))]
            else:
                v_range = [[self.minimum + 2 * x, self.minimum + 2 * (x + 1)] for x in
                           range(int((self.maximum - self.minimum) / 2))]
            logging.debug(f'v_range: {v_range}')

        return v_range

    def _calculate_disambiguation_factor(self, v, precision=False):
        '''
        determine disambiguation factor by velocity (for hypothetical phase compensation)
        NOTE: if the velocity is out of max/min bound, will return Nan.
        :param v: velocity of obeject
        :return: disambiguation factor of object
        '''
        v_range = self._velocity_range()
        reference_index = v_range.index([-1, 1])
        index_l = [x for x in range(len(v_range))]
        logging.debug(f'index list: {index_l}')
        factor_l = [x - reference_index for x in index_l]
        logging.debug(f'factor list: {factor_l}')

        result = v / self.threshold
        if precision:
            factor = result
        else:
            factor = None
            num = 0
            for v_r in v_range:
                if v_r[0] <= result < v_r[1]:
                    factor = factor_l[num]
                num += 1

        if factor is None:
            logging.error(f'Veloctiy is not correctly given or out of maximal velocity range. V = {v}')
        else:
            logging.debug(f'Factor: {factor}')

        return factor

    def convert_mat_dataset(self, mat_path, path):
        """
        convert mat data to label.csv, Xbf.npy, Xcube.npy
        :param mat_path: path of mat data
        :param path: path for saving converted data
        """
        self._creat_dir(path)
        records = os.listdir(mat_path)
        labels = self._creat_dict()
        labels_name_list = self.labels_name[3:]
        num_info = len(labels_name_list)  # number of targets's own infomations
        if self.include_Xcube:
            Xcubes_dict = {'Xcube_name': []}
        for record in records:
            if fnmatch.fnmatch(record, 'record*'):
                sub_path = os.path.join(mat_path, record)
                record_path = os.path.join(path, record)
                self._creat_dir(record_path)
                if os.path.isdir(sub_path):
                    logging.info(f'Current record folder: {record}')
                    if self.include_Xcube:
                        label, Xbfs, Xcubes = self._load_all_mats(sub_path)
                    else:
                        label, Xbfs = self._load_all_mats(sub_path)
                    labels_info = label['label'][:, 2:2 + num_info]
                    for i in range(len(labels_name_list)):
                        labels[labels_name_list[i]] = np.concatenate((labels[labels_name_list[i]], labels_info[:, i]),
                                                                     axis=None)
                        logging.debug(f"labels[labels_name_list[i]]: {len(labels[labels_name_list[i]])}")

                    for i in range(labels_info.shape[0]):
                        Xbf_name = f'{i + 1:03}.npy'
                        labels['record_name'].append(record)
                        labels['frame_name'].append(Xbf_name)
                        labels['num_target'].append(label['label'][0, 1])
                        if self.include_Xcube:
                            Xcube_name = f'Xcube_{i + 1:03}.npy'
                            Xcubes_dict['Xcube_name'].append(Xcube_name)
                            np.save(os.path.join(record_path, Xcube_name), Xcubes['permuted_Xcubes'][i])

                        np.save(os.path.join(record_path, Xbf_name), Xbfs['Xbfs'][i])

                    logging.debug(f"labels['record_name']: {labels['record_name']}")
                    logging.debug(f"labels['frame_name']: {labels['frame_name']}")
                    logging.debug(f"labels['num_target']: {labels['num_target']}")
            else:
                logging.warning(f'Please note this data (it is not record folder):  {record}')

        if self.include_Xcube:
            labels.update(Xcubes_dict)

        df = pd.DataFrame(labels)
        csv_path = os.path.join(path, 'label.csv')
        df.to_csv(csv_path, index=False, mode='a', header=not os.path.exists(csv_path))

        logging.info('Mat data convert is finished!')

    def cal_abs(self, path, save=False):
        """
        calculate abs of .npy
        """
        records = os.listdir(path)
        data_analyse = {}
        npys_abs = []
        for record in records:
            if fnmatch.fnmatch(record, 'record*'):
                logging.info(f'current folder: {record}')
                recprd_path = os.path.join(path, record)
                npys = os.listdir(recprd_path)

                for npy in npys:
                    xbf = np.load(os.path.join(recprd_path, npy))
                    logging.debug(f'shape of xbf: {xbf.shape}')
                    npys_abs.extend(abs(xbf))

        npys_abs = np.array(npys_abs)

        data_analyse['abs_max'] = npys_abs.max()
        data_analyse['abs_mean'] = npys_abs.mean()

        if save:
            df = pd.DataFrame(data_analyse, index=[1])
            csv_path = os.path.join(path, 'xbfs_abs.csv')
            df.to_csv(csv_path, index=False)
            print(f'xbfs_abs.csv is saved in {csv_path}')

        logging.info('Caculate abs of Xbfs is finished!')

        return data_analyse

    def get_all_max_abs(self, path):
        """
        get the maximal abs from all .npy
        """
        records = os.listdir(path)
        data = {}
        npys_abs = []
        for record in records:
            if fnmatch.fnmatch(record, 'record*'):
                recprd_path = os.path.join(path, record)
                npys = os.listdir(recprd_path)
                for npy in npys:
                    xbf = np.load(os.path.join(recprd_path, npy))
                    npys_abs.append(abs(xbf).max())

        npys_abs = np.array(npys_abs)
        data['max_abs'] = npys_abs
        df = pd.DataFrame(data)
        csv_path = os.path.join(path, 'all_npys_max_abs.csv')
        df.to_csv(csv_path, index=False)
        print(f'all_abs.csv is saved in {csv_path}')

    def process_decimate(self, path, start_row: int = 511, end_row: int = 561):
        """
        decimate Xbf.npy. original:1024x64 is decimate to 50x64.
        :param path: path of label.csv.
        :param start_row: first row for decimating.
        :param end_row: last row for decimating.
        """
        data = pd.read_csv(os.path.join(path, 'label.csv'))
        new_path = f'{path}_decimated'
        self._creat_dir(new_path)
        shutil.copyfile(os.path.join(path, 'label.csv'), os.path.join(new_path, 'label.csv'))
        for i in range(data.shape[0]):
            npy_path = os.path.join(path, data['record_name'].iloc[i], data['frame_name'].iloc[i])
            new_record_path = os.path.join(new_path, data['record_name'].iloc[i])
            self._creat_dir(new_record_path)
            xbf = np.load(npy_path)
            xbf = xbf[start_row:end_row]
            np.save(os.path.join(new_record_path, data['frame_name'].iloc[i]), xbf)

        if self.test_data == False:
            data_analyse = self.cal_abs(new_path)
            df = pd.DataFrame(data_analyse, index=[1])
            df.to_csv(os.path.join(new_path, 'xbfs_abs.csv'), index=False)

        print('process_decimate is finished!')

    def _process_normalize_xbf(self, xbf, data_analyse):
        """
        do normalization on xbf.npy
        """
        if isinstance(data_analyse, pd.DataFrame):
            if self.quan_norm:
                xbf = np.around(xbf * 32767 / data_analyse['abs_max'].values)
            else:
                xbf = xbf / data_analyse['abs_max'].values
        else:
            if self.quan_norm:
                xbf = np.around(xbf * 32767 / data_analyse['abs_max'])
            else:
                xbf = xbf / data_analyse['abs_max']
        return xbf

    def process_normalize(self, path, abs_path=None):
        """
        do normalization on whole dataset
        """
        data = pd.read_csv(os.path.join(path, 'label.csv'))
        if self.scaling_unit_length:
            new_path = f'{path}_normalized_scaling_unit_length'
        elif self.quan_norm:
            new_path = f'{path}_quan_normalized'
        else:
            new_path = f'{path}_normalized'

        self._creat_dir(new_path)
        shutil.copyfile(os.path.join(path, 'label.csv'), os.path.join(new_path, 'label.csv'))

        if abs_path is None:
            xbf_abs = pd.read_csv(os.path.join(path, 'xbfs_abs.csv'))
        else:
            xbf_abs = pd.read_csv(abs_path)

        for i in range(data.shape[0]):
            npy_path = os.path.join(path, data['record_name'].iloc[i], data['frame_name'].iloc[i])
            new_record_path = os.path.join(new_path, data['record_name'].iloc[i])
            self._creat_dir(new_record_path)
            xbf = np.load(npy_path)
            if self.normalize:
                if self.scaling_unit_length:
                    indices = np.nonzero(xbf)
                    xbf[indices] = xbf[indices] / abs(xbf[indices])
                else:
                    xbf = self._process_normalize_xbf(xbf, xbf_abs)

            np.save(os.path.join(new_record_path, data['frame_name'].iloc[i]), xbf)
        print('process_normalize is finished!')

    def process_doppler_vector(self, path, alpha_in=None):
        """
        get doppler vector dataset
        """
        data = pd.read_csv(os.path.join(path, 'label.csv'))
        new_path = f'{path}_doppler_vector_compressed'
        self._creat_dir(new_path)
        shutil.copyfile(os.path.join(path, 'label.csv'), os.path.join(new_path, 'label.csv'))
        for i in range(data.shape[0]):
            npy_path = os.path.join(path, data['record_name'].iloc[i], data['frame_name'].iloc[i])
            new_record_path = os.path.join(new_path, data['record_name'].iloc[i])
            self._creat_dir(new_record_path)
            xbf = np.load(npy_path)
            target_list = cfar_rect(abs(xbf), alpha_in=alpha_in)
            doppler_row = target_list[0][0]
            xbf = xbf[doppler_row]
            xbf = np.expand_dims(xbf, axis=0)
            np.save(os.path.join(new_record_path, data['frame_name'].iloc[i]), xbf)

        print('process_doppler_vector is finished!')

    def process_multiple_frames(self, path, frames_per_clip=2, step_between_clips=2, frame_3D=True):
        """
        combined serveal .npy
        function is like HMDB51 in PyTorch. https://pytorch.org/vision/main/generated
        /torchvision.datasets.HMDB51.html#torchvision.datasets.HMDB51.
        """
        if frame_3D:
            new_path = f'{path}_{frames_per_clip}frames_3D'
        else:
            new_path = f'{path}_{frames_per_clip}frames_2D'
        self._creat_dir(new_path)
        labels = self._creat_dict()
        labels_name_list = self.labels_name[3:]
        data = pd.read_csv(os.path.join(path, 'label.csv'))
        logging.debug(f'shape of data: {data.shape}')
        num_frames = len(data['frame_name'].unique())
        logging.debug(f'numbers of frames: {num_frames}')
        if frames_per_clip > num_frames or step_between_clips > num_frames:
            print(f'frames_per_clip and step_between_clips should be smaller or equal than {num_frames}!')

        left_frames = num_frames % step_between_clips
        if frames_per_clip > left_frames:
            number_clips = num_frames // step_between_clips
        else:
            number_clips = num_frames // step_between_clips + 1
        logging.debug(f'number of clips in one record: {number_clips}')

        logging.debug(int(((data.shape[0]) / num_frames)))
        median_num = int((frames_per_clip + 1) / 2 - 1)
        for i in range(int(((data.shape[0]) / num_frames))):
            record = data[self.labels_name[0]][i * num_frames]
            npys_in_one_record = data[self.labels_name[1]][i * num_frames:(i + 1) * num_frames]

            logging.info(f'Current record: {record}')
            labels_in_one_record = data[labels_name_list][i * num_frames:(i + 1) * num_frames]
            record_path = os.path.join(new_path, record)
            self._creat_dir(record_path)

            for m in range(number_clips):
                labels_clip = labels_in_one_record[
                              m * step_between_clips:m * step_between_clips + frames_per_clip]

                if frames_per_clip % 2 == 0:
                    labels_clip = np.sum(labels_clip) / len(labels_clip)
                else:
                    labels_clip = labels_clip.iloc[median_num]

                Xbf_name = f'{m + 1:03}.npy'
                labels[self.labels_name[0]].append(record)
                labels[self.labels_name[1]].append(Xbf_name)
                labels[self.labels_name[2]].append(data[self.labels_name[2]][0])
                for j in range(len(labels_name_list)):
                    if 'factor' in labels_name_list[j]:
                        factor = self._calculate_disambiguation_factor(labels_clip[labels_name_list[j - 2]])
                        labels[labels_name_list[j]].append(factor)
                    else:
                        labels[labels_name_list[j]].append(labels_clip[labels_name_list[j]])

                sub_paths = [os.path.join(path, record,
                                          npys_in_one_record[
                                          m * step_between_clips:m * step_between_clips + frames_per_clip].iloc[
                                              i])
                             for i in range(frames_per_clip)]

                Xbfs = []
                for sub_path in sub_paths:
                    logging.debug(f'Current sub_path in record: {sub_path}')
                    xbf = np.load(sub_path)
                    Xbfs.append(xbf)
                Xbfs = np.array(Xbfs)
                if frame_3D:
                    pass
                else:
                    Xbfs = Xbfs.reshape((frames_per_clip * Xbfs.shape[1], Xbfs.shape[2]))
                logging.debug(f'shape of Xbfs: {Xbfs.shape}')
                logging.debug(f'type of Xbfs: {type(Xbfs)}')

                np.save(os.path.join(record_path, Xbf_name), Xbfs)

        df = pd.DataFrame(labels)
        df.to_csv(os.path.join(new_path, 'label.csv'), index=False)

        print('process_multiple_frames is finished!')

    def process_frame_minus_frame(self, path):
        """
        do frame2-frame1, frame3-frame2...
        """
        new_path = f'{path}_delta_frames'
        self._creat_dir(new_path)

        labels = self._creat_dict()
        labels_name_list = self.labels_name[3:]
        data = pd.read_csv(os.path.join(path, 'label.csv'))
        logging.debug(f'shape of data: {data.shape}')
        num_frames = len(data['frame_name'].unique())

        for i in range(int(((data.shape[0]) / num_frames))):
            record = data[self.labels_name[0]][i * num_frames]
            npys_in_one_record = data[self.labels_name[1]][i * num_frames:(i + 1) * num_frames]

            logging.info(f'Current record: {record}')
            labels_in_one_record = data[labels_name_list][i * num_frames:(i + 1) * num_frames]
            record_path = os.path.join(new_path, record)
            self._creat_dir(record_path)

            for m in range(num_frames - 1):
                labels_clip = labels_in_one_record[m:m + 2]

                labels_clip = np.sum(labels_clip) / len(labels_clip)

                Xbf_name = f'{m + 1:03}.npy'
                labels[self.labels_name[0]].append(record)
                labels[self.labels_name[1]].append(Xbf_name)
                labels[self.labels_name[2]].append(data[self.labels_name[2]][0])
                for j in range(len(labels_name_list)):
                    if 'factor' in labels_name_list[j]:
                        factor = self._calculate_disambiguation_factor(labels_clip[labels_name_list[j - 2]])
                        labels[labels_name_list[j]].append(factor)
                    else:
                        labels[labels_name_list[j]].append(labels_clip[labels_name_list[j]])

                sub_paths = [os.path.join(path, record, npys_in_one_record[m:m + 2].iloc[i]) for i in range(2)]
                logging.debug(f'xbf path are: {sub_paths}')

                Xbfs = [np.load(sub_paths[1]) - np.load(sub_paths[0])]
                Xbfs = np.array(Xbfs)
                logging.debug(f'shape of Xbfs: {Xbfs.shape}')
                logging.debug(f'type of Xbfs: {type(Xbfs)}')

                np.save(os.path.join(record_path, Xbf_name), Xbfs)

        df = pd.DataFrame(labels)
        df.to_csv(os.path.join(new_path, 'label.csv'), index=False)

        print('process_frame_minus_frame is finished!')

    def _cfar_points(self, xbf, target, row=7, col=7):
        """
        for CFAR-ROI
        """
        if row % 2 == 0:
            print('row can only be odd')
            return

        if col % 2 == 0:
            print('col can only be odd')
            return

        row_length = row // 2
        row_midlle = row_length + 1
        col_length = col // 2
        col_midlle = col_length + 1

        if target[0] - row_length < 0:
            x1 = 0
            x2 = row
        elif target[0] + row_midlle > xbf.shape[0]:
            x1 = xbf.shape[0] - row
            x2 = xbf.shape[0]
        else:
            x1 = target[0] - row_length
            x2 = target[0] + row_midlle

        if target[1] - col_length < 0:
            y1 = 0
            y2 = col
        elif target[1] + col_midlle > xbf.shape[1]:
            y1 = xbf.shape[1] - col
            y2 = xbf.shape[1]
        else:
            y1 = target[1] - col_length
            y2 = target[1] + col_midlle

        xbf_cfar = xbf[x1:x2, y1:y2]
        return xbf_cfar

    def process_cfar_rect_single_target(self, path, row=7, col=7, scaling=False, alpha_in=None, model_path=None,
                                        threshold=0):
        """
        get CFAR-ROI dataset for 1 target
        :param row: row of ROI
        :param col: column of ROI
        :param scaling: if scaling dataset
        :param model_path: regressionmodel for scaling
        :param threshold: ab which bin will scaling be done
        """
        data = pd.read_csv(os.path.join(path, 'label.csv'))
        data['t1_range_bin'] = np.ceil(data['t1_true_range'] / 2)

        if scaling:
            new_path = f'{path}_cfar_rect_{row}x{col}_scaled_compressed'
            model = np.load(model_path)
            p = np.poly1d(model)
        else:
            new_path = f'{path}_cfar_rect_{row}x{col}_compressed'
        self._creat_dir(new_path)
        data.to_csv(os.path.join(new_path, 'label.csv'), index=False)

        for i in range(data.shape[0]):
            npy_path = os.path.join(path, data['record_name'].iloc[i], data['frame_name'].iloc[i])
            new_record_path = os.path.join(new_path, data['record_name'].iloc[i])
            self._creat_dir(new_record_path)
            xbf = np.load(npy_path)
            target_list = cfar_rect(abs(xbf), alpha_in=alpha_in)
            xbf_cfar = self._cfar_points(xbf, target_list[0], row, col)
            if scaling:
                range_bin = data['t1_range_bin'].iloc[i]
                if range_bin == threshold:
                    xbf_cfar = xbf_cfar
                else:
                    scaling_coe = p(range_bin)
                    xbf_cfar = scaling_coe * xbf_cfar
            np.save(os.path.join(new_record_path, data['frame_name'].iloc[i]), xbf_cfar)

        if self.test_data == False:
            data_analyse = self.cal_abs(new_path)
            df = pd.DataFrame(data_analyse, index=[1])
            df.to_csv(os.path.join(new_path, 'xbfs_abs.csv'), index=False)

        print('process_cfar_rect_single_target is finished!')

    def process_cfar_rect_multiple_targets(self, path, row=7, col=7, scaling=False, alpha_in=None, model_path=None,
                                           threshold=0):
        """
        get CFAR-ROI dataset for multiple target
        :param row: row of ROI
        :param col: column of ROI
        :param scaling: if scaling dataset
        :param model_path: regressionmodel for scaling
        :param threshold: ab which bin will scaling be done
        """
        labels = {}
        names = ['record_name', 'frame_name', 'classID', 'true_range_bin', 'cfar_bin', 'true_velocity', 'target',
                 'factor']
        for name in names:
            labels[name] = []
        data = pd.read_csv(os.path.join(path, 'label.csv'))
        range_list = []
        factor_list = []
        velocity_list = []
        for name in data.columns:
            if 'true_range' in name:
                range_list.append(name)
            if 'factor' in name:
                factor_list.append(name)
            if 'true_velocity' in name:
                velocity_list.append(name)

        num_target = len(range_list)
        if scaling:
            new_path = f'{path}_cfar_rect_{row}x{col}_scaled_compressed_{num_target}targets'
            model = np.load(model_path)
            p = np.poly1d(model)
        else:
            new_path = f'{path}_cfar_rect_{row}x{col}_compressed_{num_target}targets'

        self._creat_dir(new_path)

        for i in range(num_target):
            data[f't{i + 1}_range_bin'] = np.ceil(data[f't{i + 1}_true_range'] / 2)

        for i in range(data.shape[0]):
            npy_path = os.path.join(path, data['record_name'].iloc[i], data['frame_name'].iloc[i])
            new_record_path = os.path.join(new_path, data['record_name'].iloc[i])
            self._creat_dir(new_record_path)
            xbf = np.load(npy_path)
            target_list = cfar_rect(abs(xbf), alpha_in=alpha_in)
            if len(target_list) != num_target:
                print('Number of targets are not matched!')
                continue

            cfar_bins = []
            for t in target_list:
                range_bin = t[0] + 1
                cfar_bins.append(range_bin)

            if len(set(cfar_bins)) != num_target:
                print('targets are in same range bin!')
                continue

            ranges = [data[true_range].iloc[i] for true_range in range_list]
            range_bins = np.ceil(np.array(ranges) / 2)

            factors = [data[factor].iloc[i] for factor in factor_list]
            velocitys = [data[velocity].iloc[i] for velocity in velocity_list]

            r_f = {}
            for j in range(num_target):
                factor = factors[j]
                velocity = velocitys[j]
                r_f[range_bins[j]] = [factor, f't{j + 1}', velocity]
            sorted_r_f = dict(sorted(r_f.items()))

            for j in range(num_target):
                keys_in_sorted_r_f = list(sorted_r_f)
                target = target_list[j]
                cfar_bin = cfar_bins[j]
                cfar_minus_r_bins = []
                for k, v in sorted_r_f.items():
                    cfar_minus_r_bin = abs(cfar_bin - k)
                    cfar_minus_r_bins.append(cfar_minus_r_bin)
                if len(set(cfar_minus_r_bins)) != len(cfar_minus_r_bins):
                    logging.debug(
                        'target has same delta bin. can not identify with delta bin! identify with index of sorted '
                        'values!')
                    sorted_cfar_bins = sorted(cfar_bins)
                    inx = sorted_cfar_bins.index(cfar_bin)
                    inx = keys_in_sorted_r_f[inx]
                    f_t = sorted_r_f[inx]
                    target += f_t
                else:
                    inx = np.argmin(cfar_minus_r_bins)
                    inx = keys_in_sorted_r_f[inx]
                    f_t = sorted_r_f[inx]
                    target += f_t
                xbf_cfar = self._cfar_points(xbf, target, row, col)
                if scaling:
                    range_bin = data['t1_range_bin'].iloc[i]
                    if range_bin == threshold:
                        xbf_cfar = xbf_cfar
                    else:
                        scaling_coe = p(range_bin)
                        xbf_cfar = scaling_coe * xbf_cfar

                record_name = data['record_name'].iloc[i]
                frame_name = data['frame_name'].iloc[i]
                frame_name = f'{frame_name[:-4]}_{f_t[1]}.npy'
                labels['record_name'].append(record_name)
                labels['frame_name'].append(frame_name)
                labels['classID'].append(data[f'{f_t[1]}_classID'].iloc[i])
                labels['true_range_bin'].append(inx)
                labels['cfar_bin'].append(cfar_bin)
                labels['true_velocity'].append(f_t[2])
                labels['target'].append(f_t[1])
                labels['factor'].append(f_t[0])
                np.save(os.path.join(new_record_path, frame_name), xbf_cfar)

        df = pd.DataFrame(labels)
        csv_path = os.path.join(new_path, 'label.csv')
        df.to_csv(csv_path, index=False)

        if self.test_data == False:
            data_analyse = self.cal_abs(new_path)
            df = pd.DataFrame(data_analyse, index=[1])
            df.to_csv(os.path.join(new_path, 'xbfs_abs.csv'), index=False)

        print('cfar_rect_compress_multiple_targets is finished!')

    def process_cfar_rect_frame_minus_frame(self, path, alpha_in=None, row=7, col=7):
        new_path = f'{path}_cfar_rect_compressed_{row}x{col}_delta_frames'
        self._creat_dir(new_path)

        labels = self._creat_dict()
        labels_name_list = self.labels_name[3:]
        data = pd.read_csv(os.path.join(path, 'label.csv'))
        logging.debug(f'shape of data: {data.shape}')
        num_frames = len(data['frame_name'].unique())

        for i in range(int(((data.shape[0]) / num_frames))):
            record = data[self.labels_name[0]][i * num_frames]
            npys_in_one_record = data[self.labels_name[1]][i * num_frames:(i + 1) * num_frames]

            logging.info(f'Current record: {record}')
            labels_in_one_record = data[labels_name_list][i * num_frames:(i + 1) * num_frames]
            record_path = os.path.join(new_path, record)
            self._creat_dir(record_path)

            for m in range(num_frames - 1):
                labels_clip = labels_in_one_record[m:m + 2]

                labels_clip = np.sum(labels_clip) / len(labels_clip)

                Xbf_name = f'{m + 1:03}.npy'
                labels[self.labels_name[0]].append(record)
                labels[self.labels_name[1]].append(Xbf_name)
                labels[self.labels_name[2]].append(data[self.labels_name[2]][0])
                for j in range(len(labels_name_list)):
                    if 'factor' in labels_name_list[j]:
                        factor = self._calculate_disambiguation_factor(labels_clip[labels_name_list[j - 2]])
                        labels[labels_name_list[j]].append(factor)
                    else:
                        labels[labels_name_list[j]].append(labels_clip[labels_name_list[j]])

                sub_paths = [os.path.join(path, record, npys_in_one_record[m:m + 2].iloc[i]) for i in range(2)]

                Xbfs = [np.load(sub_paths[1]) - np.load(sub_paths[0])]
                Xbfs = np.array(Xbfs)
                Xbfs = np.squeeze(Xbfs)

                logging.debug(f'shape of Xbfs: {Xbfs.shape}')
                logging.debug(f'type of Xbfs: {type(Xbfs)}')

                target_list = cfar_rect(abs(Xbfs), alpha_in=alpha_in)
                abs_list = []
                for target in target_list:
                    abs_list.append(target[2])
                abs_list = np.array(abs_list)
                max_inx = np.argmax(abs_list)
                target_list = target_list[max_inx]
                Xbfs = self._cfar_points(Xbfs, target_list, row=row, col=col)
                np.save(os.path.join(record_path, Xbf_name), Xbfs)

        df = pd.DataFrame(labels)
        df.to_csv(os.path.join(new_path, 'label.csv'), index=False)

        print('process_cfar_rect_frame_minus_frame finished!')

    def balance_label_factor(self, path, classID_factor_sample_num=None):
        """
        balance dataset based on factors. factors are from HPC
        sample_num_list: how many sample should be in this factor
        """
        if classID_factor_sample_num is None:
            classID_factor = {'1': {'factors': [-2, -1, 0, 1], 'sample_num_list': [960, 480, 480, 960]},
                              '2': {'factors': [-1, 0], 'sample_num_list': [480, 480]},
                              '3': {'factors': [-1, 0], 'sample_num_list': [480, 480]}}
        data = pd.read_csv(os.path.join(path, 'label.csv'))
        data_names = data.columns
        data_list = []
        for classID, factors_sample_num_list in classID_factor.items():
            if 't1_classID' in data_names and 't2_classID' not in data_names:
                data_by_ID = data[data['t1_classID'] == int(classID)]
            else:
                data_by_ID = data[data['classID'] == int(classID)]
            logging.debug(f'classID is {classID}')
            factors = factors_sample_num_list['factors']
            sample_num_list = factors_sample_num_list['sample_num_list']

            for i in range(len(factors)):
                if 't1_classID' in data_names and 't2_classID' not in data_names:
                    data_by_ID_velocityrange = data_by_ID[data_by_ID['t1_factor'] == factors[i]]
                else:
                    data_by_ID_velocityrange = data_by_ID[data_by_ID['factor'] == factors[i]]
                sample_num = sample_num_list[i]
                if data_by_ID_velocityrange.shape[0] < sample_num:
                    print(f'tagert type {classID} and factor {factors[i]} do NOT have enough frames for data balance')
                    print(f'Number of sample needed: {sample_num}, current has: {data_by_ID_velocityrange.shape[0]}')
                    return
                logging.debug(f'Number of sample: {sample_num} for tagert type {classID} and factor {factors[i]}')
                logging.debug(f'shape of data before random sample: {data_by_ID_velocityrange.shape}')
                data_by_ID_velocityrange = data_by_ID_velocityrange.sample(n=sample_num)
                logging.debug(f'shape of data after random sample: {data_by_ID_velocityrange.shape}')
                data_list.append(data_by_ID_velocityrange)
        balanced_data = pd.concat(data_list)
        logging.debug(f'shape of new label.csv: {balanced_data.shape}')
        csv_path = os.path.join(path, 'balanced_label.csv')
        balanced_data.to_csv(csv_path, index=False)
        print('balance_label_factor is finished')

    def balance_label_velocity(self, csv_path, save_path, threshold=18.24, velocity_sample_num=None):
        """
        balance dataset based on velocity range.
        threshold = Vmax
        ['-4 to -3', '-3 to -2', '-2 to -1', '-1 to 0','0 to 1','1 to 2']Vmax is converted in [-3,-2,-1,0,1,2]:
        '-4 to -3' is -3, '-3 to -2' is -2...

        sample_num_list: how many samples should be for this velocity range
        """
        if velocity_sample_num is None:
            classID_velocity = {
                '1': {'velocityrange': [-3, -2, -1, 0, 1, 2], 'sample_num_list': [200, 50, 50, 50, 50, 200]},
                '2': {'velocityrange': [-2, -1, 0, 1], 'sample_num_list': [50, 50, 50, 50]},
                '3': {'velocityrange': [-1, 0, 1], 'sample_num_list': [100, 50, 50]}}
        data = pd.read_csv(csv_path)
        data.drop_duplicates(subset=['record_name', 'frame_name'], inplace=True)
        data_names = data.columns
        if 't1_classID' in data_names and 't2_classID' not in data_names:
            print('1 target')
            velocity = np.ceil(data['t1_true_velocity'] / threshold)
        else:
            print('multiple targets')
            velocity = np.ceil(data['true_velocity'] / threshold)
        data_list = []
        for classID, velocityrange_sample_num_list in classID_velocity.items():
            if 't1_classID' in data_names and 't2_classID' not in data_names:
                data_by_ID = data[data['t1_classID'] == int(classID)]
            else:
                data_by_ID = data[data['classID'] == int(classID)]
            logging.debug(f'classID is {classID}')
            velocityranges = velocityrange_sample_num_list['velocityrange']
            sample_num_list = velocityrange_sample_num_list['sample_num_list']

            for i in range(len(velocityranges)):
                data_by_ID_velocityrange = data_by_ID[velocity == velocityranges[i]]
                sample_num = sample_num_list[i]
                if data_by_ID_velocityrange.shape[0] < sample_num:
                    print(
                        f'tagert type {classID} and velocity range {velocityranges[i]} do NOT have enough frames for data balance')
                    print(f'Number of sample needed: {sample_num}, current has: {data_by_ID_velocityrange.shape[0]}')
                    return
                logging.debug(
                    f'Number of sample: {sample_num} for tagert type {classID} and factor {velocityranges[i]}')
                logging.debug(f'shape of data before random sample: {data_by_ID_velocityrange.shape}')
                data_by_ID_velocityrange = data_by_ID_velocityrange.sample(n=sample_num)
                logging.debug(f'shape of data after random sample: {data_by_ID_velocityrange.shape}')
                data_list.append(data_by_ID_velocityrange)
        balanced_data = pd.concat(data_list)
        balanced_data.reset_index()
        logging.debug(f'shape of new label.csv: {balanced_data.shape}')
        csv_path = os.path.join(save_path, 'balanced_label.csv')
        balanced_data.to_csv(csv_path, index=False)
        print('balance_label_velocity is finished')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    preparedata = PrepareDataset()

    preparedata.normalize = True
    preparedata.scaling_unit_length = False
    preparedata.quan_norm = True
    preparedata.test_data = False
    # if mat data has Xcube, please set this as True
    preparedata.include_Xcube = False

    # labels_name for 1 target is default, no need to set. if for multiple target, example for 2 targets is:
    # labels_name for 2 targets
    # labels_name = ['record_name', 'frame_name', 'num_target', 't1_true_range',
    #                't1_det_range', 't1_true_velocity', 't1_det_velocity', 't1_factor', 't1_classID', 't2_true_range',
    #                't2_det_range', 't2_true_velocity', 't2_det_velocity', 't2_factor', 't2_classID']
    # preparedata.labels_name = labels_name

    # labels_name = ['record_name', 'frame_name', 'num_target', 't1_true_range', 't1_true_velocity','t1_classID']
    # preparedata.labels_name = labels_name
    # mat_path = 'mat_rxNF=17'
    # save_path = 'rxNF=17'
    # preparedata.convert_mat_dataset(mat_path, save_path)
    # preparedata.process_decimate(save_path)
    preparedata.process_normalize('rxNF=34_decimated', 'dataset_1_target/1000records_compressed/xbfs_abs.csv')
    preparedata.process_normalize('rxNF=51_decimated', 'dataset_1_target/1000records_compressed/xbfs_abs.csv')


