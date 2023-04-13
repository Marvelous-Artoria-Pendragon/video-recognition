import os
from sklearn.model_selection import train_test_split

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
# from mypath import Path


class VideoDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.

        Args:
            dataset (str):                  Name of dataset.
            split (str):                    Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            clip_len (int):                 Determines how many frames are there in each clip, which must be less than min_split_freq. Defaults to 16.
            preprocess (bool):              Determines whether to preprocess dataset. Default is False.
            resize_height (int):            Resize the picture's height. Defaults to 128.
            resize_height (int):            Resize the picture's width. Defaults to 171.
            crop_size (int):                The size of random crop.
            label_path (str):               The diretory of label.
            test_size (float):              The split size of test data among the whole dataset.
            val_size (float):               The split size of valid data among the train dataset.
            min_split_freq (int):           The minimum number of split frames. Default is 24.
    """

    def __init__(self, dataset, split='train', clip_len=16, preprocess=False, resize_height = 128, resize_width = 171, crop_size = 112, 
                 label_dir = './dataloader',  root_dir = '.', output_dir = '.', test_size = 0.2, val_size = 0.2, min_frames = 24, norm = 'minus'):
        self.root_dir = root_dir
        self.output_dir = output_dir
        folder = os.path.join(self.output_dir, split)
        self.clip_len = clip_len
        self.split = split
        self.norm = norm

        # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.crop_size = crop_size
        self.test_size = test_size
        self.val_size = val_size
        self.min_frames = min_frames

        if not self.check_integrity():
            print(self.root_dir)
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You need to download it from official website.')

        if (not self.check_preprocess()) or preprocess:
            print('Preprocessing of {} dataset, this will take long, but it will be done only once.'.format(dataset))
            self.preprocess()

        # Obtain all the filenames of files inside all the class folders
        # Going through each class folder one at a time
        self.fnames, labels = [], []
        for label in sorted(os.listdir(folder)):
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                labels.append(label)

        assert len(labels) == len(self.fnames)
        print('Number of {} videos: {:d}'.format(split, len(self.fnames)))

        # Prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        # Convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        label_filename = os.path.join(label_dir, dataset + '_labels.txt')
        if not os.path.exists(label_filename):
            with open(label_filename, 'w') as f:
                for id, label in enumerate(sorted(self.label2index)):
                    f.writelines(str(id+1) + ' ' + label + '\n')

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # Loading and preprocessing.
        buffer = self.load_frames(self.fnames[index])
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        labels = np.array(self.label_array[index])

        if self.split == 'test':
            # Perform data augmentation
            buffer = self.randomflip(buffer)
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        return torch.from_numpy(buffer), torch.from_numpy(labels)

    def check_integrity(self):
        return os.path.exists(self.root_dir)

    def check_preprocess(self):
        # TODO: Check image size in output_dir
        if not os.path.exists(self.output_dir):
            return False
        elif not os.path.exists(os.path.join(self.output_dir, 'train')):
            return False

        for ii, video_class in enumerate(os.listdir(os.path.join(self.output_dir, 'train'))):
            for video in os.listdir(os.path.join(self.output_dir, 'train', video_class)):
                video_name = os.path.join(os.path.join(self.output_dir, 'train', video_class, video),
                                    sorted(os.listdir(os.path.join(self.output_dir, 'train', video_class, video)))[0])
                image = cv2.imread(video_name)
                if np.shape(image)[0] != self.resize_height or np.shape(image)[1] != self.resize_width:
                    return False
                else:
                    break

            if ii == 10:
                break

        return True

    def preprocess(self):
        if not os.path.exists(self.output_dir):
            # os.mkdir(self.output_dir)
            os.makedirs(self.output_dir)
            os.mkdir(os.path.join(self.output_dir, 'train'))
            os.mkdir(os.path.join(self.output_dir, 'val'))
            os.mkdir(os.path.join(self.output_dir, 'test'))

        # Split train/val/test sets
        for file in os.listdir(self.root_dir):
            file_path = os.path.join(self.root_dir, file)
            video_files = [name for name in os.listdir(file_path)]

            train_and_valid, test = train_test_split(video_files, test_size = self.test_size, random_state=42)
            train, val = train_test_split(train_and_valid, test_size = self.val_size, random_state=42)

            train_dir = os.path.join(self.output_dir, 'train', file)
            val_dir = os.path.join(self.output_dir, 'val', file)
            test_dir = os.path.join(self.output_dir, 'test', file)

            print(train_dir)
            if not os.path.exists(train_dir):
                os.makedirs(train_dir)
            if not os.path.exists(val_dir):
                os.makedirs(val_dir)
            if not os.path.exists(test_dir):
                os.makedirs(test_dir)

            for video in train:
                self.process_video(video, file, train_dir)

            for video in val:
                self.process_video(video, file, val_dir)

            for video in test:
                self.process_video(video, file, test_dir)

        print('Preprocessing finished.')

    def process_video(self, video, action_name, save_dir):
        # Initialize a VideoCapture object to read video data into a numpy array
        video_filename = video.split('.')[0]
        if not os.path.exists(os.path.join(save_dir, video_filename)):
            os.mkdir(os.path.join(save_dir, video_filename))

        capture = cv2.VideoCapture(os.path.join(self.root_dir, action_name, video))

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # extract frame equidistantly
        EXTRACT_FREQUENCY = 4
        while (EXTRACT_FREQUENCY > 1 and frame_count // EXTRACT_FREQUENCY <= self.min_frames):
            EXTRACT_FREQUENCY -= 1

        count = 0; L = []; i = 0
        retaining = True
        frame = None
        while (count < frame_count and retaining):
            retaining, frame = capture.read()
            if frame is None:
                continue

            if count % EXTRACT_FREQUENCY == 0:
                if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                L.append(frame)
            count += 1

        # when the frames is less than min_frames, loop the video
        while len(L) < self.min_frames:
            L.append(L[i])
            i = (i + 1) % count

        for ii in range(len(L)):
            cv2.imwrite(filename=os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(ii))), img=L[ii])

        # Release the VideoCapture once it is no longer needed
        capture.release()

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer


    def normalize(self, buffer):
        norms = {'minus': lambda x: (x/255.0) * 2 - 1, 'fix': lambda x: x - np.array([[[90.0, 98.0, 102.0]]]),
                 'mm': lambda x: (x-np.min(x) / (np.max(x)-np.min(x))), 'z-score': lambda x: (x-np.mean(x)) / (np.sqrt((np.sum(x-np.mean(x)**2))/(x.shape[0] * x.shape[1]))),
                 'log': lambda x: np.log10(x) / np.log10(np.max(x)), 'atan': lambda x: np.arctan(x)*(2/np.pi)}
        for i, frame in enumerate(buffer):
            # frame -= np.array([[[90.0, 98.0, 102.0]]])  # the method mentioned in C3D source code
            # frame = (frame/255.0) * 2 - 1
            frame = norms[self.norm](frame)
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[i] = frame

        return buffer
        
    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        # print('buffer shape: ', buffer.shape)
        time_index = np.random.randint(buffer.shape[0] - clip_len + 1)

        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    train_data = VideoDataset(dataset='ucf101', split='test', clip_len=8, preprocess=False)
    train_loader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=4)

    for i, sample in enumerate(train_loader):
        inputs = sample[0]
        labels = sample[1]
        print(inputs.size())
        print(labels)

        if i == 1:
            break