import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

import argparse
from dataloaders.dataset import VideoDataset
from network import C3D_model, R2Plus1D_model, R3D_model, I3D_model, Resnet, Boosting_model

# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

class_dict = dict({'hmdb51': 51, 'ucf101':101, 'tb':14, 'test':2})

# save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

def train_model(dataset, modelName, n_classes, save_dir = '.', lr = 1e-3, nEpochs = 50, model_dir = None, data_dir = None, label_dir = './dataloader', 
                resume_epoch = 0, snapshots = 10, useTest = True, test_interval = 10, num_frames = 16, resize_height = 128, resize_width = 171, 
                crop_size = 112, batch_size = 12, n_worker = 4, modality = 'rgb', pretrained = False, step_size = 10, gamma = 0.1):
    """
    ===========================================================================\n
        Args:
            modelName (str): C3D, RPlus1D, R3D, Resnet(series, e.g: Resnet18, Resnet50), I3D.
            n_classes (int): Number of classes in the data.
            nEpochs (int, optional): Number of epochs to train for.
            lr (float): Learning rate.
            resume_epoch (int): Resume previous training from the epoch.
            useTest (bool): See evolution of the test set when training.
            test_interval (int): Run on the test set every nTestInterval epochs.
            num_frames (int): Number of the frames of each clip.
    ===========================================================================\n
    """

    if modelName == 'C3D':
        model = C3D_model.C3D(num_classes=n_classes, pretrained = pretrained, model_path = os.path.join(model_dir, "c3d-pretrained.pth"))
        train_params = [{'params': C3D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': C3D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    elif modelName == 'R2Plus1D':
        model = R2Plus1D_model.R2Plus1DClassifier(num_classes=n_classes, layer_sizes=(2, 2, 2, 2))
        train_params = [{'params': R2Plus1D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': R2Plus1D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    elif modelName == 'R3D':
        model = R3D_model.R3DClassifier(num_classes=n_classes, layer_sizes=(2, 2, 2, 2))
        train_params = model.parameters()
    elif len(modelName) > 6 and modelName[:6] == 'Resnet':     # Resnet
        model = Resnet.generate_model(int(modelName[6:]), n_classes = n_classes)
        train_params = model.parameters()
    elif modelName == 'I3D':
        # default rgb mode
        if modality == 'rgb':
            model = I3D_model.InceptionI3d(400, num_frames=num_frames, in_channels=3)
            pretrained_path = os.path.join(model_dir, "rgb_imagenet.pt")
        else:   # flow
            model = I3D_model.InceptionI3d(400, num_frames=num_frames, in_channels=2)
            pretrained_path = os.path.join(model_dir, "flow_imagenet.pt")
        if pretrained:
                model.load_state_dict(torch.load(pretrained_path))
        model.replace_logits(n_classes)
        train_params = model.parameters()
    else:
        print('We only implemented C3D, R2Plus1D, Resnet seris and I3D models.')
        raise NotImplementedError
    criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
    optimizer = optim.SGD(train_params, lr = lr, momentum = 0.9, weight_decay = 5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = gamma)  # the scheduler divides the lr by 1/gamma every update epochs

    if resume_epoch == 0:
        print("Training {} from scratch...".format(modelName))
    else:
        checkpoint = torch.load(os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar'),
                       map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
        print("Initializing weights from: {}...".format(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    # if modelName == 'I3D':
    model = nn.DataParallel(model)
    model.to(device)
    criterion.to(device)

    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    train_dataloader = DataLoader(VideoDataset(dataset=dataset, split='train',clip_len=num_frames, resize_height = resize_height, resize_width = resize_width, crop_size = crop_size, 
                                               label_dir = label_dir, root_dir = data_dir, output_dir = save_dir), batch_size = batch_size, shuffle=True, num_workers=n_worker)
    val_dataloader   = DataLoader(VideoDataset(dataset=dataset, split='val',  clip_len=num_frames, resize_height = resize_height, resize_width = resize_width, crop_size = crop_size, 
                                               label_dir = label_dir, root_dir = data_dir, output_dir = save_dir), batch_size = batch_size, num_workers=n_worker)
    test_dataloader  = DataLoader(VideoDataset(dataset=dataset, split='test', clip_len=num_frames, resize_height = resize_height, resize_width = resize_width, crop_size = crop_size, 
                                               label_dir = label_dir, root_dir = data_dir, output_dir = save_dir), batch_size = batch_size, num_workers=n_worker)

    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)

    print('Training model on {} dataset...'.format(dataset))
    for epoch in range(resume_epoch, nEpochs):
        # each epoch has a training and validation step
        for phase in ['train', 'val']:
            start_time = timeit.default_timer()

            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0.0

            # set model to train() or eval() mode depending on whether it is trained
            # or being validated. Primarily affects layers such as BatchNorm or Dropout.
            if phase == 'train':
                model.train()
            else:
                model.eval()

            for inputs, labels in tqdm(trainval_loaders[phase]):
                # move inputs and labels to the device the training is taking place on
                inputs = Variable(inputs, requires_grad=True).to(device)
                labels = Variable(labels).to(device)
                optimizer.zero_grad()

                if phase == 'train':
                    outputs = model(inputs)
                else:
                    with torch.no_grad():
                        outputs = model(inputs)

                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels.long())

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / trainval_sizes[phase]
            epoch_acc = running_corrects.double() / trainval_sizes[phase]

            if phase == 'train':
                writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/train_acc_epoch', epoch_acc, epoch)
            else:
                writer.add_scalar('data/val_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/val_acc_epoch', epoch_acc, epoch)

            print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch+1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

        # scheduler.step() is to be called once every epoch during training
        scheduler.step()

        if epoch % snapshots == (snapshots - 1):
            if modelName == 'I3D':
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.module.state_dict(),
                    'opt_dict': optimizer.state_dict(),
                }, os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar'))
            else:
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'opt_dict': optimizer.state_dict(),
                }, os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar'))
            print("Save model at {}\n".format(os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar')))

        if useTest and epoch % test_interval == (test_interval - 1):
            model.eval()
            start_time = timeit.default_timer()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in tqdm(test_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    outputs = model(inputs)
                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels.long())

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / test_size
            epoch_acc = running_corrects.double() / test_size

            writer.add_scalar('data/test_loss_epoch', epoch_loss, epoch)
            writer.add_scalar('data/test_acc_epoch', epoch_acc, epoch)

            print("[test] Epoch: {}/{} Loss: {} Acc: {}".format(epoch+1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

    writer.close()

def dataset_distribution(root_dir, output_dir = '.'):
    x = []; y = []
    for cla_name in sorted(os.listdir(root_dir)):
        cla_path = os.path.join(root_dir, cla_name)
        x.append(cla_name)
        y.append(len(os.listdir(cla_path)))
    plt.figure(figsize=(10,6))
    plt.bar(range(len(y)), y)
    plt.xticks(range(len(y)), x, rotation = 30 )
    plt.xlabel('Class Name')
    plt.ylabel('Number of Samples')
    plt.title('Sample Distribution')
    plt.savefig(os.path.join(output_dir, 'dataset_distribution.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("A Combination of C3D, R3D, R2Plus1D, I3D, Resnet on video recognition.\n")
    parser.add_argument('--model',                  type = str, default = 'C3D', help = "C3D, R2Plus1D, R3D, Resnet+depth(e.g: Resnet18), I3D, GBDT, XGBoost.")
    parser.add_argument('--epoch',                  type = int, default = 40, help = "Number of epochs for training. Default 40.")
    parser.add_argument('--resume_epoch',           type = int, default = 0, help = "Default 0, change if want to resume.")
    parser.add_argument('--useTest',                action = "store_true", help = "See evolution of the test set when training. Default true.")
    parser.add_argument('--nTestInterval',          type = int, default = 10, help = "Run on the test set every nTestInterval epochs. Default 10.")
    parser.add_argument('--snapshot',               type = int, default = 10, help = "Store a model every snapshot epochs. Default 10.")
    parser.add_argument('--lr',                     type = float, default = 1e-3, help = "Learning rate, default 1e-3.")
    parser.add_argument('--n_frame',                type = int, default = 16, help = "Number of the frames of each clip. Default 16.")
    parser.add_argument('--dataset',                type = str, default = "ucf101", help = "The name of dataset (hmd51, ucf101), other dataset please modify mypath.py. Default ucf101.")
    parser.add_argument('--data_dir',               type = str, default = ".", help = "The path of the dataset.")
    parser.add_argument('--save_dir',               type = str, default = ".", help = "The path to save the output. Default is current diretory.")
    parser.add_argument('--model_dir',              type = str, default = ".", help = "The path of pretrained model. Default is current diretory.")
    parser.add_argument('--label_dir',              type = str, default = "./dataloader", help = "The path of labels. Defaul is ./dataloader .")
    parser.add_argument('--n_class',                type = int, default = 0, help = "The number of class.")
    parser.add_argument('--height',                 type = int, default = 128, help = "The resize height of video. Default 128.")
    parser.add_argument('--width',                  type = int, default = 171, help = "The resize width of video. Default 172.")
    parser.add_argument('--crop_size',              type = int, default = 112, help = "The centre crop size of video. Default 112.")
    parser.add_argument('--batch_size',             type = int, default = 12, help = "The batch size of training, validation and test. Default 12.")
    parser.add_argument('--n_worker',               type = int, default = 4, help = "The number of cpu that use to load data. Default is 4.")
    parser.add_argument('--modality',               type = str, default = 'rgb', help = "The mode of video, rgb or flow which determines the number of channels. Default is rgb.")
    parser.add_argument('--pretrained',             action = 'store_true', help = "Use pretrained model if possible. Default False.")
    parser.add_argument('--step_size',              type = int, default = 10 , help = "How many steps to update learning rate. Default 10.")
    parser.add_argument('--step_gamma',             type = float, default = 0.1 , help = "The scheduler times the lr by step_gamma each update step. Default 0.1.")
    parser.add_argument('--draw',                   action = "store_true", help = "Draw the distribution of dataset, which is saved in the path '--save_dir_root'.")
    parser.add_argument('--feature_path',           type = str, help = "The path of feature. It works when use GBDT and xgboost")
    parser.add_argument('--n_estimators',           type = int, default = 100, help = 'The quantity of DT trees. Default is 100.')
    parser.add_argument('--n_iter',                 type = int, default = 0, help = 'Early stopping after n_iter if score is no change. It works when the model is GBDT. Default is None.')
    parser.add_argument('--loss',                   type = str, default = 'deviance', help = "The loss function of GBDT to optimize. ('deviance', 'exponential')")
    parser.add_argument('--min_samples_split',      type = int, default = 2, help = "The minimum samples of trees to split in one node. Default 2.")
    parser.add_argument('--min_samples_leaf',       type = int, default = 1, help = "The minimum samples of the terminal node in one tree. Default 1.")
    parser.add_argument('--max_depth',              type = int, default = 3, help = "The max depth of trees. Default 3.")
    parser.add_argument('--gamma',                  type = float, default = 0.1, help = "Post-pruning (XGBoost, 0.1~0.2 recommended). Default is 0.1.")
    parser.add_argument('--n_thread',               type = int, default = 1, help = "The number of CPU threads that used to train XGBoost. Default is 1.")
    parser.add_argument('--eta',                    type = float, default = 0.007, help = "Similar to learing rate (XGBoost). Default is 0.007")

    args = parser.parse_args()
    saveName = args.model + '-' + args.dataset
    n_classes = class_dict.get(args.dataset, 0)
    if n_classes == 0:
        if not args.n_class:
            print("Cannot match the existed dataset or missing num_class arguement!")
            raise ValueError
        else:
            n_classes = args.n_class
    if args.n_iter <= 0:
        n_iter = None
    if args.model == 'GBDT':
        Boosting_model.GBDT(feature_path = args.feature_path,
                            lr = args.lr,
                            n_estimators = args.n_estimators,
                            n_iter_no_change = args.n_iter,
                            model_save_path = args.save_dir,
                            save_name = saveName,
                            loss = args.loss,
                            min_samples_split = args.min_samples_split,
                            min_samples_leaf = args.min_samples_leaf,
                            max_depth = args.max_depth)
    elif args.model == 'XGBoost':
        Boosting_model.XGBoost(feature_path = args.feature_path,
                                num_classes = args.n_class, 
                                lr = args.lr,
                                max_depth = args.max_depth,
                                n_estimators = args.n_estimators,
                                gamma = args.gamma,
                                n_thread = args.n_thread,
                                eta = args.eta,
                                model_save_path = args.save_dir,
                                save_name = saveName)
    else:
        if args.resume_epoch != 0:
            runs = sorted(glob.glob(os.path.join(args.save_dir, 'run', 'run_*')))
            run_id = int(runs[-1].split('_')[-1]) if runs else 0
        else:
            runs = sorted(glob.glob(os.path.join(args.save_dir, 'run', 'run_*')))
            run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

        save_dir = os.path.join(args.save_dir, 'run', 'run_' + str(run_id))
        train_model(dataset = args.dataset,
                    modelName = args.model,
                    n_classes = n_classes,
                    save_dir = args.save_dir,
                    lr = args.lr,
                    nEpochs = args.epoch,
                    model_dir = args.model_dir,
                    data_dir = args.data_dir,
                    label_dir = args.label_dir,
                    resume_epoch = args.resume_epoch,
                    snapshots = args.snapshot,
                    useTest = args.useTest,
                    test_interval = args.nTestInterval,
                    num_frames = args.n_frame,
                    resize_height = args.height,
                    resize_width = args.width,
                    crop_size = args.crop_size,
                    batch_size = args.batch_size,
                    n_worker = args.n_worker,
                    modality = args.modality,
                    pretrained = args.pretrained,
                    step_size = args.step_size,
                    gamma = args.step_gamma
                    )
    if args.draw:
        dataset_distribution(args.data_dir, args.save_dir)