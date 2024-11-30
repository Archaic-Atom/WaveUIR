import os
import numpy as np
import argparse
import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
import lightning.pytorch as pl

from utils.dataset_utils import DenoiseTestDataset, DerainDehazeDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
from net.WaveUIR import WaveUIR

class WaveUIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = WaveUIR(dim = 32, num_blocks = [2,3,3,4], num_refinement_blocks = 2, task_num=testopt.task_num)
        self.loss_fn  = nn.L1Loss()

    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)
        loss = self.loss_fn(restored,clean_patch)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=180)

        return [optimizer],[scheduler]


def test_Denoise(net, dataset, sigma=15):
    output_path = testopt.output_path + 'denoise/' + str(sigma) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])
    
    dataset.set_sigma(sigma)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([clean_name, task_id], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored = net(degrad_patch)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)

            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            save_image_tensor(restored, output_path + clean_name[0] + "_%f" % (temp_psnr) + '.png')

        print("Denoise sigma=%d: psnr: %.2f, ssim: %.4f" % (sigma, psnr.avg, ssim.avg))
    return psnr.avg


def test_Derain_Dehaze(net, dataset, task="derain"):
    output_path = testopt.output_path + task + '/'
    subprocess.check_output(['mkdir', '-p', output_path])

    dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([degraded_name, task_id], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored = net(degrad_patch)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(restored, output_path + degraded_name[0] + "_%f" % (temp_psnr) + '.png')
        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))
    return psnr.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--mode', type=int, default=6,
                        help='0 for denoise, 1 for derain, 2 for dehaze, 3 for deblur, 4 for enhance, 5 for all-in-one (three tasks), 6 for all-in-one (five tasks)')
    
    parser.add_argument('--task_num', type=int, default=5, help='task num')
    parser.add_argument('--gopro_path', type=str, default="/data/zzd/Datasets/UIR/WaveUIR/test/deblur/", help='save path of test hazy images')
    parser.add_argument('--enhance_path', type=str, default="/data/zzd/Datasets/UIR/WaveUIR/test/enhance/", help='save path of test hazy images')
    parser.add_argument('--denoise_path', type=str, default="/data/zzd/Datasets/UIR/WaveUIR/test/denoise/", help='save path of test noisy images')
    parser.add_argument('--derain_path', type=str, default="/data/zzd/Datasets/UIR/WaveUIR/test/derain/", help='save path of test raining images')
    parser.add_argument('--dehaze_path', type=str, default="/data/zzd/Datasets/UIR/WaveUIR/test/dehaze/", help='save path of test hazy images')

    parser.add_argument('--output_path', type=str, default="WaveUIR_Visresults/", help='output save path')
    parser.add_argument('--ckpt_name', type=str, default="WaveUIR_5d.ckpt", help='checkpoint save path')
    testopt = parser.parse_args()
    
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(testopt.cuda)

    ckpt_path = "ckpt/" + testopt.ckpt_name

    if testopt.task_num == 1:
        denoise_splits = ["bsd68/", "urban100/"]
    else:
        denoise_splits = ["bsd68/"]
    derain_splits = ["Rain100L/"]
    deblur_splits = ["gopro/"]
    enhance_splits = ["lol/"]

    denoise_tests = []
    derain_tests = []

    base_path = testopt.denoise_path
    for i in denoise_splits:
        testopt.denoise_path = os.path.join(base_path,i)
        denoise_testset = DenoiseTestDataset(testopt)
        denoise_tests.append(denoise_testset)

    print("CKPT name : {}".format(ckpt_path))

    net  = WaveUIRModel().load_from_checkpoint(ckpt_path).cuda()
    net.eval()

    if testopt.mode == 0:
        for testset,name in zip(denoise_tests,denoise_splits) :
            print('Start {} testing Sigma=15...'.format(name))
            test_Denoise(net, testset, sigma=15)

            print('Start {} testing Sigma=25...'.format(name))
            test_Denoise(net, testset, sigma=25)

            print('Start {} testing Sigma=50...'.format(name))
            test_Denoise(net, testset, sigma=50)

    elif testopt.mode == 1:
        print('Start testing rain streak removal...')
        derain_base_path = testopt.derain_path
        for name in derain_splits:
            print('Start testing {} rain streak removal...'.format(name))
            testopt.derain_path = os.path.join(derain_base_path,name)
            derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=15)
            test_Derain_Dehaze(net, derain_set, task="derain")

    elif testopt.mode == 2:
        print('Start testing SOTS...')
        derain_base_path = testopt.derain_path
        name = derain_splits[0]
        testopt.derain_path = os.path.join(derain_base_path,name)
        derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=15)
        test_Derain_Dehaze(net, derain_set, task="dehaze")

    elif testopt.mode == 3:
        print('Start testing GOPRO...')
        deblur_base_path = testopt.gopro_path
        name = deblur_splits[0]
        testopt.gopro_path = os.path.join(deblur_base_path,name)
        derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=15, task='deblur')
        test_Derain_Dehaze(net, derain_set, task="deblur")

    elif testopt.mode == 4:
        print('Start testing LOL...')
        enhance_base_path = testopt.enhance_path
        name = enhance_splits[0]
        testopt.enhance_path = os.path.join(enhance_base_path,name)
        derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=15, task='enhance')
        test_Derain_Dehaze(net, derain_set, task="enhance")

    elif testopt.mode == 5:
        total_psnr = 0.0
        for testset,name in zip(denoise_tests,denoise_splits) :
            print('Start {} testing Sigma=15...'.format(name))
            denoise15_psnr = test_Denoise(net, testset, sigma=15)
            total_psnr += denoise15_psnr

            print('Start {} testing Sigma=25...'.format(name))
            denoise25_psnr = test_Denoise(net, testset, sigma=25)
            total_psnr += denoise25_psnr

            print('Start {} testing Sigma=50...'.format(name))
            denoise50_psnr = test_Denoise(net, testset, sigma=50)
            total_psnr += denoise50_psnr

        derain_base_path = testopt.derain_path
        print(derain_splits)
        for name in derain_splits:

            print('Start testing {} rain streak removal...'.format(name))
            testopt.derain_path = os.path.join(derain_base_path,name)
            derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=55)
            derain_psnr = test_Derain_Dehaze(net, derain_set, task="derain")
            total_psnr += derain_psnr

        print('Start testing SOTS...')
        dehaze_psnr = test_Derain_Dehaze(net, derain_set, task="dehaze")
        total_psnr += dehaze_psnr

        print("mean psnr : %.2f" % (total_psnr / 5.0))

    elif testopt.mode == 6:
        total_psnr = 0.0
        for testset,name in zip(denoise_tests,denoise_splits) :
            print('Start {} testing Sigma=25...'.format(name))
            denoise_psnr = test_Denoise(net, testset, sigma=25)
            total_psnr += denoise_psnr

        derain_base_path = testopt.derain_path
        print(derain_splits)
        for name in derain_splits:

            print('Start testing {} rain streak removal...'.format(name))
            testopt.derain_path = os.path.join(derain_base_path,name)
            derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=55)
            derain_psnr = test_Derain_Dehaze(net, derain_set, task="derain")
            total_psnr += derain_psnr

        print('Start testing SOTS...')
        dehaze_psnr = test_Derain_Dehaze(net, derain_set, task="dehaze")
        total_psnr += dehaze_psnr

        deblur_base_path = testopt.gopro_path
        for name in deblur_splits:
            print('Start testing GOPRO...')

            # print('Start testing {} rain streak removal...'.format(name))
            testopt.gopro_path = os.path.join(deblur_base_path,name)
            deblur_set = DerainDehazeDataset(testopt,addnoise=False,sigma=55, task='deblur')
            deblur_psnr = test_Derain_Dehaze(net, deblur_set, task="deblur")
            total_psnr += deblur_psnr

        enhance_base_path = testopt.enhance_path
        for name in enhance_splits:

            print('Start testing LOL...')
            testopt.enhance_path = os.path.join(enhance_base_path,name)
            derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=55, task='enhance')
            enhance_psnr = test_Derain_Dehaze(net, derain_set, task="enhance")
            total_psnr += enhance_psnr
        print("mean psnr : %.2f" % (total_psnr / 5.0))
