'''
从本地获取图像数据
'''
import os,random
import numpy as np
from PIL import Image
import pickle
import time
import uuid

def resize_image(in_image, new_width, new_height, out_image=None,
                 resize_mode=Image.ANTIALIAS):
    """ Resize an image.

    Arguments:
        in_image: `PIL.Image`. The image to resize.
        new_width: `int`. The image new width.
        new_height: `int`. The image new height.
        out_image: `str`. If specified, save the image to the given path.
        resize_mode: `PIL.Image.mode`. The resizing mode.

    Returns:
        `PIL.Image`. The resize image.

    """
    img = in_image.resize((new_width, new_height), resize_mode)
    if out_image:
        img.save(out_image)
    return img



def to_categorical(y, nb_classes):
    """ to_categorical.

    Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy.

    Arguments:
        y: `array`. Class vector to convert.
        nb_classes: `int`. Total number of classes.

    """
    y = np.asarray(y, dtype='int32')
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    Y[np.arange(len(y)),y] = 1.
    return Y

def shuffle(*arrs):
    """ shuffle.

    Shuffle given arrays at unison, along first axis.

    Arguments:
        *arrs: Each array to shuffle at unison.

    Returns:
        Tuple of shuffled arrays.

    """
    arrs = list(arrs)
    for i, arr in enumerate(arrs):
        assert len(arrs[0]) == len(arrs[i])
        arrs[i] = np.array(arr)
    p = np.random.permutation(len(arrs[0]))
    return tuple(arr[p] for arr in arrs)


def get_img_channel(image_path):
    """
    Load a image and return the channel of the image
    :param image_path:
    :return: the channel of the image
    """
    img = load_image(image_path)
    img = pil_to_nparray(img)
    try:
        channel = img.shape[2]
    except:
        channel = 1
    return channel

def ListPic(dir,fileNameList,samples,targets,label,flags=None,filter_channel=False):
    for sample in fileNameList:
        if not flags or any(flag in sample for flag in flags):
            if filter_channel:
                if get_img_channel(os.path.join(dir, sample)) != 3:
                    continue
            samples.append(os.path.join(dir, sample))
            targets.append(label)

def ListDir(dir,dirList,samples,targets,label,flags=None,filter_channel=False):
    for dirName in dirList:
        dirPath = os.path.join(dir,dirName)
        walk = os.walk(dirPath).__next__()
        ListDir(dir=dirPath, dirList=walk[1], samples=samples, targets=targets, label=label, flags=flags,
                filter_channel=filter_channel)
        ListPic(dir=dirPath, fileNameList=walk[2], samples=samples, targets=targets, label=label, flags=flags,
                filter_channel=filter_channel)

def directory_to_samples(directory, flags=None, filter_channel=False):
    """ Read a directory, and list all subdirectories files as class sample """
    samples = []
    targets = []
    label = 0
    classes = sorted(os.walk(directory).__next__()[1])
    for c in classes:
        c_dir = os.path.join(directory, c)
        walk = os.walk(c_dir).__next__()
        ListDir(dir=c_dir,dirList=walk[1],samples=samples,targets=targets,label=label,flags=flags,filter_channel=filter_channel)
        ListPic(dir=c_dir,fileNameList=walk[2],samples=samples,targets=targets,label=label,flags=flags,filter_channel=filter_channel)
        label += 1
    return samples, targets

def load_image(in_image):
    """ Load an image, returns PIL.Image. """
    img = Image.open(in_image)
    return img

def convert_color(in_image, mode):
    """ Convert image color with provided `mode`. """
    return in_image.convert(mode)


def pil_to_nparray(pil_image):
    """ Convert a PIL.Image to numpy array. """
    pil_image.load()
    return np.asarray(pil_image, dtype="float32")


def image_dirs_to_samples(directory, resize=None, convert_gray=None,
                          filetypes=None):
    print("Starting to parse images...")
    if filetypes:
        if filetypes not in [list, tuple]: filetypes = list(filetypes)
    samples, targets = directory_to_samples(directory, flags=filetypes)
    print('Sample is ok,convert image')
    print("samples number is %d " %(len(samples)))
    for i, s in enumerate(samples):
        print(s)
        samples[i] = load_image(s)
        if resize:
            samples[i] = resize_image(samples[i], resize[0], resize[1])
        if convert_gray:
            samples[i] = convert_color(samples[i], 'L')
        samples[i] = pil_to_nparray(samples[i])
        samples[i] /= 255.

    print("Parsing Done!")
    return samples, targets


def build_image_dataset_from_dir(directory,
                                 dataset_file="my_tflearn_dataset.pkl",
                                 resize=None, convert_gray=None,
                                 filetypes=None, shuffle_data=False,
                                 categorical_Y=False):
    try:
        X, Y = pickle.load(open(dataset_file, 'rb'))
    except Exception:
        X, Y = image_dirs_to_samples(directory, resize, convert_gray, filetypes)

        if categorical_Y:
            Y = to_categorical(Y, np.max(Y) + 1) # First class is '0'
        if shuffle_data:
            X, Y = shuffle(X, Y)
        pickle.dump((X, Y), open(dataset_file, 'wb'))
    return X, Y

#dirname 根目录
#imagefolder 图片目录
#pklname 训练数据
def load_data(dirname="TrainData",
              imagefolder="gray",
              pklname='ctimage.pkl',
              convert_gray=True,
              resize_pics=(227, 227), shuffle=True,one_hot=False):
    dataset_file = os.path.join(dirname, pklname)
    if not os.path.exists(dataset_file):
        imagePath = os.path.join(dirname,imagefolder)
        if not os.path.exists(imagePath):
            raise Exception("%s doesn't exist " %(imagePath))

    X, Y = build_image_dataset_from_dir(os.path.join(dirname,imagefolder),
                                        dataset_file=dataset_file,
                                        resize=resize_pics,
                                        filetypes=['.jpg', '.jpeg'],
                                        convert_gray=convert_gray,
                                        shuffle_data=shuffle,
                                        categorical_Y=one_hot)
    X = np.asarray(X, dtype=np.float32)
    if convert_gray:
        X = X.reshape([-1,resize_pics[0],resize_pics[1],1])
    else:
        X = X.reshape([-1, resize_pics[0], resize_pics[1], 3])

    X_train, X_val = X[:-5000], X[-5000:]
    y_train, y_val = Y[:-5000], Y[-5000:]
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    X_val = np.asarray(X_val,dtype=np.float32)
    y_val = np.asarray(y_val,dtype=np.int64)
    return X_train,y_train,X_val,y_val



#############################################华丽的分割线#############################################

def load_pkl(pklpath,resize_pics,convert_gray):
    X, Y = pickle.load(open(pklpath, 'rb'))
    X = np.asarray(X, dtype=np.float32)
    if convert_gray:
        X = X.reshape([-1,resize_pics[0],resize_pics[1],1])
    else:
        X = X.reshape([-1, resize_pics[0], resize_pics[1], 3])

    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.int64)
    return X,Y

#判断pkl后缀的文件是否存在
#ext 用于判断是否存在预测的样本数据,如果ext = .pkl 不存在样本数据时候，用imagedata生成pkl后缀文件；如果是其他值，仅仅提示文件不存在
def load_mul_data(dirname="TrainData",
              imagefolder="imagedata",
              ext='.pkl',
              filetypes=['.jpg', '.jpeg'],
              convert_gray=False,
              resize_pics=(227, 227), shuffle_data=True,one_hot=False):
    pkls = []
    walk = os.walk(dirname).__next__()
    for file in walk[2]:
        if file[file.rfind('.'):len(file)] == ext:
            pkls.append(os.path.join(dirname,file))

    if ext == '.pkl':
        if len(pkls) == 0:
            #
            print("Starting to sample images...")
            if filetypes:
                if filetypes not in [list, tuple]: filetypes = list(filetypes)

            X, Y = directory_to_samples(os.path.join(dirname, imagefolder), flags=filetypes)
            print('Sample is ok,')
            print("samples number is %d " % (len(X)))
            if one_hot:
                Y = to_categorical(Y, np.max(Y) + 1)  # First class is '0'
            if shuffle_data:
                X, Y = shuffle(X, Y)

            samples, targets = [], []
            predict = False  #作为预测数据
            for i, s in enumerate(X):
                print(s)
                if i > 0 and i % 2500 == 0:
                    samplePath = os.path.join(dirname, str(uuid.uuid1()) + '.pkl')
                    if predict == False:
                        samplePath += '.predict'
                        predict = True
                    else:
                        pkls.append(samplePath)
                    print('保存数据 %s' %(samplePath))
                    pickle.dump((samples, targets), open(samplePath, 'wb'))

                    samples, targets = [], []
                samples.append(load_image(s))
                if resize_pics:
                    samples[-1] = resize_image(samples[-1], resize_pics[0], resize_pics[1])
                if convert_gray:
                    samples[-1] = convert_color(samples[-1], 'L')
                samples[-1] = pil_to_nparray(samples[-1])
                samples[-1] /= 255.
                targets.append(Y[i])

            if len(samples) > 0:
                samplePath = os.path.join(dirname, str(uuid.uuid1()) + '.pkl')
                print('保存数据 %s' % (samplePath))
                pickle.dump((samples, targets),open( samplePath, 'wb'))
                pkls.append(samplePath)
            print("Sample image Done!")
    else:
        if len(pkls) == 0:
            raise Exception("%s目录下 %s后缀文件不存在,无法评估" % (dirname,ext))
    return pkls