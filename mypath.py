'''
This file edit the paths of models, pre-trained weight, labels and input/output (dataset)
'''

class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'ucf101':
            # folder that contains class labels
            root_dir = '/kaggle/input/ucf101/UCF101/UCF-101'
            output_dir = '/kaggle/working/ucf101'
            return root_dir, output_dir
            
            # Save preprocess data into output_dir
            # root_dir = './UCF-101'
            # output_dir = './ucf101'

        elif database == 'hmdb51':
            # folder that contains class labels
            root_dir = '/Path/to/hmdb-51'

            output_dir = '/path/to/VAR/hmdb51'

            return root_dir, output_dir
        elif database == 'tb':
            root_dir = '/kaggle/input/cibr-14/CIBR-14'
            output_dir = '/kaggle/working/CIBR-14'
            return root_dir, output_dir
        
            # root_dir = './tb-8'
            # output_dir = './tb-8'
            # return root_dir, output_dir
        elif database == 'test':
            root_dir = './CIBR-2'
            output_dir = './out'
            return root_dir, output_dir
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        # return '/path/to/Models/c3d-pretrained.pth'
        return '/kaggle/input/c3d-pretrained/c3d-pretrained.pth'

    @staticmethod
    def ucf_label_dir():
        return '/kaggle/input/c3d-model/dataloaders/ucf_labels.txt'
        # return 'dataloaders/tests_labels.txt'
    
    @staticmethod
    def tb_label_dir():
        # return '/kaggle/input/c3d-model/dataloaders/tb_labels.txt'
        return '/kaggle/input/video-recognition/dataloaders/tb_labels.txt'
        # return 'dataloaders/tb_labels.txt'
    
    @staticmethod
    def rgb_imagenet_dir():
        # return '/kaggle/input/rgb-imagenet-pt/rgb_imagenet.pt'
        return './models/rgb_imagenet.pt'