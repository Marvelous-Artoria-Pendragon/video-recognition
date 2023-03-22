import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataloaders.dataset import VideoDataset
from network import C3D_model, R2Plus1D_model, R3D_model, I3D_model, Resnet
import argparse

# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

class_dict = dict({'hmdb51': 51, 'ucf101':101, 'tb':14, 'test':2})

def extract_feature(dataset, model_path, num_classes, model_name = 'I3D', num_frames = 16, modality = 'rgb', save_dir = '.', 
                    resize_height = 226, resize_width = 226, crop_size = 224, batch_size = 12, num_worker = 0, label_path = './dataloader'):
    if model_name == 'I3D':
        if modality == 'flow':
            model = I3D_model.InceptionI3d(num_classes, in_channels=2, num_frames = num_frames)
        else:
            model = I3D_model.InceptionI3d(num_classes, in_channels=3, num_frames = num_frames)
    else:
        print('We only implemented I3D models.')
        raise NotImplementedError
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)

    train_dataloader = DataLoader(VideoDataset(dataset=dataset, split='train',clip_len=num_frames, resize_height=resize_height, resize_width=resize_width, crop_size = crop_size), 
                                  batch_size=batch_size, shuffle=True, num_workers=num_worker, label_path = label_path)
    val_dataloader   = DataLoader(VideoDataset(dataset=dataset, split='val',  clip_len=num_frames, resize_height=resize_height, resize_width=resize_width, crop_size = crop_size), 
                                  batch_size=batch_size, num_workers=num_worker, label_path = label_path)
    test_dataloader  = DataLoader(VideoDataset(dataset=dataset, split='test', clip_len=num_frames, resize_height=resize_height, resize_width=resize_width, crop_size = crop_size), 
                                  batch_size=batch_size, num_workers=num_worker, label_path = label_path)
    
    print('Extracting feature with {} on {} dataset...'.format(model_name, dataset))
    data_loader = dict({'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader})
    model.train(False)
    for name, loader in data_loader.items():
        features = []
        labels = []
        for inputs, label in tqdm(loader):
            # get the inputs

            labels.append(label.data.cpu().numpy())
            b,c,t,h,w = inputs.shape
            if t > 1600:    # Only extract features for 1600 frames when it is more than 1600 frames.
                for start in range(1, t-56, 1600):
                    end = min(t-1, start+1600+56)
                    start = max(1, start-48)
                    ip = Variable(torch.from_numpy(inputs.numpy()[:,:,start:end]).cuda(), volatile=True)
                    output = model.extract_features(ip)
                    features.append(output.squeeze(2).squeeze(2).squeeze(2).data.cpu().numpy())
                
            else:
                # wrap them in Variable
                inputs = Variable(inputs.cuda(), volatile=True)
                output = model.extract_features(inputs)
                features.append(output.squeeze(2).squeeze(2).squeeze(2).data.cpu().numpy())
        # feature: (time, height, width, channel)
        np.save(os.path.join(save_dir, name + '_' + model_name + '_' + dataset + '_' + 'data'), np.vstack(features))
        np.save(os.path.join(save_dir, name + '_' + model_name + '_' + dataset + '_' + 'label'), np.hstack(labels))

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Extract feature with I3D.\n")
    parser.add_argument('--dataset',        type = str, help = "Dataset's name ('hmd51', 'ucf101', or custom dataset).")
    parser.add_argument('--n_class',        type = int, default = 0, help = "The number of class.")
    parser.add_argument('--model_path',     type = str, help = "Model's path (pt file, now only support I3D).")
    parser.add_argument('--modality',       type = str, default = 'rgb', help = "'rgb' (3 channels) or 'flow' (2 channels).")
    parser.add_argument('--num_frame',      type = int, default = 16, help = "The number of frames per second.")
    parser.add_argument('--model',          type = str, default = 'I3D', help = "The name of model (now I3D only).")
    parser.add_argument('--save_dir',       type = str, default = '.', help = "The path to save the features. Default is current diretory.")
    parser.add_argument('--batch_size',     type = int, default = 12, help = "Batch size of data loader.")
    parser.add_argument('--num_worker',     type = int, default = 0, help = "The number of cpu to load data.")
    parser.add_argument('--label_path',     type = str, default = './dataloader', help = "The path of label.")
    parser.add_argument('--height',         type = int, default = 226, help = "The resize height of video. Default 226.")
    parser.add_argument('--width',          type = int, default = 226, help = "The resize width of video. Default 226.")
    parser.add_argument('--crop_size',      type = int, default = 224, help = "The centre crop resize video. Default 224.")

    args = parser.parse_args()
    num_classes = class_dict.get(args.data, 0)
    if num_classes == 0:
        if not args.n_class:
            print("Cannot match the existed dataset or missing num_class arguement!")
            raise ValueError
        else:
            n_classes = args.n_class
    if not args.data:
        print('Data is null.')
        raise ValueError
    elif not args.model_path:
        print("The path of model is not provided.")
        raise ValueError
    else:
        extract_feature(dataset = args.dataset, 
                        model_path = args.model_path, 
                        num_frames = args.num_frame,
                        modality = args.modality,
                        save_dir = args.save_dir,
                        resize_height = args.height,
                        resize_width = args.width,
                        crop_size = args.crop_size,
                        batch_size = args.batch_size,
                        num_worker = args.num_worker,
                        label_path = args.label_path)