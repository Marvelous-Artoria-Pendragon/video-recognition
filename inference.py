import torch
import numpy as np
from network import C3D_model, R2Plus1D_model, R3D_model, I3D_model, Resnet
import cv2
torch.backends.cudnn.benchmark = True
import os
import argparse
import joblib
from tqdm import tqdm

def CenterCrop(frame, size):
    h, w = np.shape(frame)[0:2]
    th, tw = size
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))

    frame = frame[y1:y1 + th, x1:x1 + tw, :]
    return np.array(frame).astype(np.uint8)


def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)

def inference(modelName, num_classes, check_point, label_path, extractor_path, video_dir = '.', output_dir = './out', modality = 'rgb', num_frames = 16, 
              resize_height = 226, resize_width = 226, crop_size = 224, visual = False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    with open(label_path, 'r') as f:
        class_names = f.readlines()
        f.close()
    extractor = None
    # init model
    if modelName == 'C3D':
        model = C3D_model.C3D(num_classes = num_classes)
    elif modelName == 'R2Plus1D':
        model = R2Plus1D_model.R2Plus1DClassifier(num_classes = num_classes, layer_sizes=(2, 2, 2, 2))
    elif modelName == 'R3D':
        model = R3D_model.R3DClassifier(num_classes = num_classes, layer_sizes=(2, 2, 2, 2))
    elif modelName == 'I3D':
        if modality == 'rgb':
            model = I3D_model.InceptionI3d(num_classes, in_channels = 3, num_frames = num_frames)
        else:   # flow
            model = I3D_model.InceptionI3d(num_classes, in_channels = 2, num_frames = num_frames)
    elif len(modelName) > 6 and modelName[:6] == 'Resnet':
        model = Resnet.generate_model(int(modelName[6:]), n_classes = num_classes)
    if modelName == 'GBDT' or modelName == 'XGBoost':
        if modality == 'rgb':
            extractor = I3D_model.InceptionI3d(num_classes, in_channels = 3, num_frames = num_frames)
        else:   # flow
            extractor = I3D_model.InceptionI3d(num_classes, in_channels = 2, num_frames = num_frames)
        cpt = torch.load(extractor_path, map_location=lambda storage, loc: storage)
        extractor.load_state_dict(cpt['state_dict'])
        extractor.to(device)
        model = joblib.load(check_point)
    else:
        checkpoint = torch.load(check_point, map_location = lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # read video
    print('Reading videoes...')
    pbar = tqdm(sorted(os.listdir(video_dir)))
    for video in pbar:
        if len(video.split('.')) < 2:       # directory
            continue
        pbar.set_description('Processing %s' % video)
        name, fm = video.split('.')      # video name and format
        if fm == 'mp4' or fm == 'avi':
            cap = cv2.VideoCapture(os.path.join(video_dir, video))
            retaining = True

            clip = []
            # save the predict video with tag
            save_path = os.path.join(output_dir, name + '_pred_' + modelName + '.avi')

            four_cc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            freq = cap.get(cv2.CAP_PROP_FPS)
            video_writer = cv2.VideoWriter(save_path, four_cc, float(freq), size)

            while retaining:
                retaining, frame = cap.read()
                if not retaining and frame is None:
                    continue
                # tmp_ = center_crop(cv2.resize(frame, (171, 128)))
                # tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
                tmp_ = CenterCrop(cv2.resize(frame, (resize_width, resize_height)), (crop_size, crop_size))
                tmp = (tmp_/255.) * 2 - 1
                clip.append(tmp)
                if len(clip) == num_frames:
                    inputs = np.array(clip).astype(np.float32)
                    inputs = np.expand_dims(inputs, axis=0)
                    inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
                    inputs = torch.from_numpy(inputs)
                    inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)
                    if extractor:
                        output = extractor.extract_features(inputs)
                        feature = output.squeeze(2).squeeze(2).squeeze(2).data.cpu().numpy()
                        # probs = model.predict_proba(feature)[:,1]
                        probs = model.predict_proba(feature)
                        label = np.argmax(probs[0])

                    else:
                        with torch.no_grad():
                            outputs = model.forward(inputs)

                        probs = torch.nn.Softmax(dim=1)(outputs)
                        label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]

                    cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (20, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 255), 1)
                    cv2.putText(frame, "prob: %.4f" % probs[0][label], (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 255), 1)
                    clip.pop(0)
                video_writer.write(frame)
                # remove the annaotation below, if you want to see the result
                if visual:
                    cv2.imshow('result', frame)
                    cv2.waitKey(50)

            video_writer.release()
            cap.release()
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("This program inference the video for each frames with the given model.\n")
    parser.add_argument('--model',              type = str, default = 'C3D', help = "C3D, R2Plus1D, R3D, Resnet(e.g:Resnet18, Resnet34..), I3D, GBDT, XGBoost")
    parser.add_argument('--n_class',            type = int, default = 0, help = "The number of class.")
    parser.add_argument('--check_point',        type = str, help = "Path of model's checkpoint (pt/pkl file).")
    parser.add_argument('--label_path',         type = str, help = "The path of label (txt file).")
    parser.add_argument('--video_dir',          type = str, default = '.', help = "The diretory of orginal videoes.")
    parser.add_argument('--output_dir',         type = str, default = './out', help = "The diretory of inferenced videoes.")
    parser.add_argument('--modality',           type = str, default = 'rgb', help = "The mode of video, rgb or flow which determines the number of channels. Default is rgb.")
    parser.add_argument('--n_frame',            type = int, default = 16, help = "Number of the frames of each clip. Default 16.")
    parser.add_argument('--visual',             action = 'store_true', help = "Show the result with cv.")
    parser.add_argument('--height',             type = int, default = 226, help = "The resize height of video. Default 226.")
    parser.add_argument('--width',              type = int, default = 226, help = "The resize width of video. Default 226.")
    parser.add_argument('--crop_size',          type = int, default = 224, help = "The centre crop size of video. Default 224.")
    parser.add_argument('--extractor',          type = str, help = "The I3D model (default) to extract feature, if use GBDT/XGBoost.")
    
    args = parser.parse_args()
    if args.n_class <= 0:
        print("Cannot match the existed dataset or missing num_class arguement!")
        raise ValueError
    elif not args.check_point:
        print("Missing argument: 'check_point'. ")
    elif not args.label_path:
        print("Missing argument: 'label_path'. ")
    inference(modelName = args.model,
              num_classes = args.n_class,
              check_point = args.check_point,
              extractor_path = args.extractor,
              label_path = args.label_path,
              video_dir = args.video_dir,
              output_dir = args.output_dir,
              modality = args.modality,
              num_frames = args.n_frame,
              visual = args.visual)

    # inference(modelName = 'XGBoost',
    #           num_classes = 14,
    #           check_point = 'models/XGBoost-CIBR-14',
    #           extractor_path = 'models/I3D-tb_epoch-19_norm_without_pretrained.pth.tar',
    #           label_path = 'dataloaders/tb_labels.txt',
    #           video_dir = 'open-course/test-video',
    #           output_dir = 'open-course/test-out',
    #           modality = 'rgb',
    #           num_frames = 16,
    #           visual = False)
# python inference.py --model GBDT --n_class 14 --check_point models/gbdt_frame24.pkl --label_path dataloaders/tb_labels.txt --video_dir open-course --output_dir /open-course/output --extractor models/I3D-tb_epoch-19_norm_without_pretrained.pth.tar --n_frame 16
# python inference.py --model I3D --n_class 14 --check_point models/I3D-tb_epoch-39_frame24.pth.tar --label_path dataloaders/tb_labels.txt --video_dir open-course --output_dir /open-course/output  --n_frame 24







