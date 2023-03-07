import torch
import numpy as np
from network import C3D_model, R2Plus1D_model, R3D_model, I3D_model, Resnet
import cv2
torch.backends.cudnn.benchmark = True
import os
import uuid

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

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    with open('./dataloaders/tb_labels.txt', 'r') as f:
        class_names = f.readlines()
        f.close()
    # init model
    # model = C3D_model.C3D(num_classes=8)
    # model = R2Plus1D_model.R2Plus1DClassifier(num_classes=8, layer_sizes=(2, 2, 2, 2))
    # model = Resnet.generate_model(34, n_classes = 14)
    model = I3D_model.InceptionI3d(400)
    # checkpoint = torch.load('run/run_1/models/C3D_ucf101_epoch-39.pth.tar', map_location=lambda storage, loc: storage)
    # checkpoint = torch.load('/kaggle/input/resnet34-tb-epoch-49/ResNet34-tb_epoch-49.pth.tar', map_location=lambda storage, loc: storage)
    checkpoint = torch.load('.\models\I3D-tb_epoch-49.pth.tar', map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    # read video
    # video = 'F:\\High level language programming\\python\\Teaching Behavior Classification\\hmdb51\\shoot_ball\\Clay_sBasketballSkillz_shoot_ball_f_nm_np1_ba_med_2.avi'
    # video = r'.\tb-8\stps\I_1s_stps_0004_03.avi'
    # for file in os.listdir('/kaggle/input/open-course'):
    # root_path = os.getcwd()
    video = r'.\open-course\course_01.mp4'
    # video = os.path.join(root_path, file)
    cap = cv2.VideoCapture(video)
    retaining = True

    clip = []
    # save the predict video with tag
    save_path = './open-course/course_01_pred_I3D.avi'
    # save_path = os.path.join('/kaggle/working/' + file + '_pred_Resnet34.avi', )

    four_cc = cv2.VideoWriter_fourcc(*'XVID')
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    freq = cap.get(cv2.CAP_PROP_FPS)
    video_writer = cv2.VideoWriter(save_path, four_cc, float(freq), size)

    while retaining:
        retaining, frame = cap.read()
        if not retaining and frame is None:
            continue
        tmp_ = center_crop(cv2.resize(frame, (171, 128)))
        tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
        clip.append(tmp)
        if len(clip) == 16:
            inputs = np.array(clip).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
            inputs = torch.from_numpy(inputs)
            inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)
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
        # cv2.imshow('result', frame)
        # cv2.waitKey(50)

    video_writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()









