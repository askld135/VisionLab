import numpy as np
import struct

def read_bin_file(filepath):
    '''
        read '.bin' file to 2-d numpy array

    :param path_bin_file:
        path to '.bin' file

    :return:
        2-d image as numpy array (float32)

    '''

    data = np.fromfile(filepath, dtype=np.uint16)
    ww, hh = data[:2]

    data_2d = data[2:].reshape((hh, ww))
    data_2d = data_2d.astype(np.float32) 

    return data_2d

def save_bin(filepath, arr):
    '''
        save 2-d numpy array to '.bin' files with uint16

    @param filepath:
        expected file path to store data

    @param arr:
        2-d numpy array

    @return:
        None

    '''

    arr = np.round(arr).astype('uint16')
    arr = np.clip(arr, 0, 1023)
    height, width = arr.shape

    with open(filepath, 'wb') as fp:
        fp.write(struct.pack('<HH', width, height))
        arr.tofile(fp)

if __name__ == '__main__':

    filepath = './sample/rgbw_001_fullres_0db.bin'
    rgbw_data = read_bin_file(filepath)

    savepath = './sample/rgbw_001_fullres_0db_saved.bin'
    save_bin(savepath, rgbw_data)