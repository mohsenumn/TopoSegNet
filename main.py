import torch
import torch.nn as nn
from torchvision.utils import save_image
import numpy as np
from skimage.io import imshow, imread, imsave
import argparse, json
import os, glob, sys
from time import time
from dataloader import DRIVE
from unet.unet_model import UNet
from PIL import Image
from utilities import torch_dice_fn_bce
from topoloss_td import TopologicalLoss, CustSigmoid
import matplotlib.pyplot as plt
import shutil

torch.cuda.empty_cache()

def parse_func(args):
    print("Reading params file {}...".format(args.params))
    with open(args.params, 'r') as f:
        params = json.load(f)

    task = params['common']['task']
    mydict = {}
    mydict['num_classes'] = int(params['common']['num_classes'])
    mydict['folders'] = [params['common']['img_folder'], params['common']['gt_folder']]
    mydict["checkpoint_restore"] = params['common']['checkpoint_restore']

    if task == "train":
        mydict['train_datalist'] = params['train']['train_datalist']
        mydict['validation_datalist'] = params['train']['validation_datalist']
        mydict['output_folder'] = params['train']['output_folder']
        mydict['crop_size'] = params['train']['crop_size']
        mydict['train_batch_size'] = int(params['train']['train_batch_size'])
        mydict['val_batch_size'] = int(params['train']['val_batch_size'])
        mydict['learning_rate'] = float(params['train']['learning_rate'])
        mydict['num_epochs'] = int(params['train']['num_epochs']) + 1
        mydict['save_every'] = params['train']['save_every']
        mydict['topo_weight'] = params['train']['topo_weight'] # If 0 => not training with topoloss
        mydict['topo_window'] = params['train']['topo_window']
        mydict['output_suffix'] = params['train']['output_suffix']

    elif task == "test":
        mydict['test_datalist'] = params['test']['test_datalist']
        mydict['output_folder'] = params['test']['output_folder']

    else:
        print("Wrong task chosen")
        sys.exit()


    if not os.path.exists(mydict['output_folder']):
        os.makedirs(mydict['output_folder'])

    print(task, mydict)
    return task, mydict

def set_seed(): 
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))
     
def train_2d(mydict):
    set_seed()
    device = torch.device("cuda")
    print("CUDA device: {}".format(device))

    if not torch.cuda.is_available():
        print("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")
    
    force_cudnn_initialization()
    
    # Train Data       
    training_set = DRIVE(mydict['train_datalist'], mydict['folders'], task="train", crop_size=mydict['crop_size'])
    training_generator = torch.utils.data.DataLoader(training_set, batch_size=mydict['train_batch_size'], shuffle=True, num_workers=4, drop_last=True)
    
    # Validation Data
    validation_set = DRIVE(mydict['validation_datalist'], mydict['folders'], task="val", crop_size=mydict['crop_size'])
    validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=mydict['val_batch_size'], shuffle=False, num_workers=2, drop_last=False)
    
    # Network
    network = UNet(n_channels=3, n_classes=mydict['num_classes'], start_filters=64).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=mydict['learning_rate'], weight_decay=0)
    
    # Load checkpoint (if specified)
    if mydict['checkpoint_restore'] != "":
        network.load_state_dict(torch.load(mydict['checkpoint_restore']), strict=True)
        print("loaded checkpoint! {}".format(mydict['checkpoint_restore']))
    
    # Losses
    bce_loss_func = torch.nn.BCELoss(size_average = False, reduce=False, reduction=None)
    topo_loss_func = TopologicalLoss()
    
    # Train loop
    best_dict = {}
    best_dict['epoch'] = 0
    best_dict['val_loss'] = None
    print("Let the training begin!")
    num_batches = len(training_generator)
    cussig = CustSigmoid(alpha=10, beta=0.5)
    for epoch in range(mydict['num_epochs']):
        torch.cuda.empty_cache() 
        network.to(device).train() 
        avg_train_loss = 0.0
        epoch_start_time = time()
        for step, (patch, mask, _) in enumerate(training_generator): 
            t_los = 0
            optimizer.zero_grad()
            patch = patch.to(device, dtype=torch.float)
            mask = mask.to(device, dtype=torch.float)
            y_pred = torch.sigmoid(network(patch))  
            loss_val = torch.mean(bce_loss_func(y_pred, mask)) 

            # Ours
            if mydict['topo_weight'] != 0:
                topo_loss = 0
                a = topo_loss_func(cussig(y_pred), mask, num_erosions=3, mode='thin')
                topo_loss += a[0]
                topo_loss = topo_loss * mydict['topo_weight']
                t_los += topo_loss
            else:
                t_los = 0
                
            total_loss = loss_val + t_los
            avg_train_loss += loss_val
            total_loss.backward()
            optimizer.step()
            
        print('topo_loss= ', t_los)
        print('   ') 
        avg_train_loss /= num_batches
        epoch_end_time = time()
        print("Epoch {} \nAverage training loss: {}".format(epoch, avg_train_loss))
        print("Epoch {} took {} seconds.\nAverage training loss: {}".format(epoch, epoch_end_time-epoch_start_time, avg_train_loss))
        validation_start_time = time()
        with torch.no_grad():
            network.eval()
            validation_iterator = iter(validation_generator)
            avg_val_loss = 0.0
            for _ in range(len(validation_generator)):
                x, y_gt, _ = next(validation_iterator)
                x = x.to(device, non_blocking=True)
                y_gt = y_gt.to(device, non_blocking=True)
                y_pred = torch.sigmoid(network(x))
                avg_val_loss += torch_dice_fn_bce(y_pred, y_gt)
            avg_val_loss /= len(validation_generator)
        validation_end_time = time()
        print("Average validation loss: {}".format(avg_val_loss))
        print("End of epoch validation took {} seconds.\nAverage validation loss: {}".format(validation_end_time - validation_start_time, avg_val_loss))
        # check for best epoch and save it if it is and print
        if epoch == 0:
            best_dict['epoch'] = epoch
            best_dict['val_loss'] = avg_val_loss
        elif best_dict['val_loss'] > avg_val_loss:
                best_dict['val_loss'] = avg_val_loss
                best_dict['epoch'] = epoch
                torch.save(network.state_dict(), os.path.join(mydict['output_folder'], f"model_best_{mydict['output_suffix']}.pth"))
        print("Best epoch so far: {}\n".format(best_dict))
        # save checkpoint for save_every
        if epoch % mydict['save_every'] == 0:
            torch.save(network.state_dict(), os.path.join(mydict['output_folder'], "model_epoch" + str(epoch) + f"_{mydict['output_suffix']}.pth"))

def test_2d(mydict):
    device = torch.device("cuda")
    print("CUDA device: {}".format(device))

    if not torch.cuda.is_available():
        print("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

    # Test Data
    test_set = DRIVE(mydict['test_datalist'], mydict['folders'], task="test", crop_size=mydict['crop_size'])
    n_channels = 3

    test_generator = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2, drop_last=False)

    network = UNet(n_channels=n_channels, n_classes=mydict['num_classes'], start_filters=64).to(device)

    if mydict['checkpoint_restore'] != "":
        network.load_state_dict(torch.load(mydict['checkpoint_restore']), strict=True)
        print("loaded checkpoint! {}".format(mydict['checkpoint_restore']))
    else:
        print("No model found!")
        sys.exit()

    print("Let the inference begin!")
    print("Todo: {}".format(len(test_generator)))

    with torch.no_grad():
        network.eval()
        test_iterator = iter(test_generator)
        for _ in range(len(test_generator)):
            x, y_gt, filename = next(test_iterator)
            x = x.to(device, non_blocking=True)
            y_gt = y_gt.to(device, non_blocking=True)

            y_pred = torch.sigmoid(network(x))

            filename = filename[0]
            save_image(x, os.path.join(mydict['output_folder'], 'img_' + filename ))
            save_image(torch.squeeze(y_gt*255), os.path.join(mydict['output_folder'], 'gt_' + filename)) # can be used since using BCE with num_classes=1

            np_pred = torch.squeeze(y_pred).detach().cpu().numpy()
            np_pred = np.where(np_pred >= 0.5, 1., 0.) # 0.5 thresholding
            np_pred = (np_pred*255.).astype(np.uint8)
            im_pred = Image.fromarray(np_pred)
            im_pred.save(os.path.join(mydict['output_folder'], 'pred_' + filename))

            print("{} Done!".format(filename))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, help="Path to the JSON parameters file")
    
    if len(sys.argv) == 1:
        print("Path to parameters file not provided. Exiting...")

    else:
        args = parser.parse_args()
        task, mydict = parse_func(args)

    if task == "train":
        train_2d(mydict)
    elif task == "test":
        test_2d(mydict)

