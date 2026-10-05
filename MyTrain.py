import os
import logging
import numpy as np
from datetime import datetime
from tensorboardX import SummaryWriter
from Net.MyNet import  MyNet as Network
from torch.cuda.amp import autocast , GradScaler
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import optim
from torchvision.utils import make_grid
import eval.python.metrics as Measure
from utils.clip_grad import clip_gradient
from utils.dataset import get_loader, test_dataset

def cal_ual(seg_logits, seg_gts):
    assert seg_logits.shape == seg_gts.shape, (seg_logits.shape, seg_gts.shape)
    sigmoid_x = seg_logits.sigmoid()
    loss_map = 1 - (2 * sigmoid_x - 1).abs().pow(2)
    return loss_map.mean()

def get_coef(iter_percentage, method):
    if method == "linear":
        milestones = (0.3, 0.7)
        coef_range = (0, 1)
        min_point, max_point = min(milestones), max(milestones)
        min_coef, max_coef = min(coef_range), max(coef_range)
        if iter_percentage < min_point:
            ual_coef = min_coef
        elif iter_percentage > max_point:
            ual_coef = max_coef
        else:
            ratio = (max_coef - min_coef) / (max_point - min_point)
            ual_coef = ratio * (iter_percentage - min_point)
    elif method == "cos":
        coef_range = (0, 1)
        min_coef, max_coef = min(coef_range), max(coef_range)
        normalized_coef = (1 - np.cos(iter_percentage * np.pi)) / 2
        ual_coef = normalized_coef * (max_coef - min_coef) + min_coef
    else:
        ual_coef = 1.0
    return ual_coef




def structure_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / (weit.sum(dim=(2, 3))+ 1e-4)

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()

def train(train_loader, model, optimizer, epoch, save_path, writer,scaler):
    """
    train function
    """
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.cuda()
            gts = gts.cuda()

            with autocast():
                p1,pcnn,p1_B,T_Loss = model(images)

                loss_p1 = structure_loss(p1, gts)
                loss_pcnn = structure_loss(pcnn, gts)
                loss_pred = loss_p1 + loss_pcnn

                loss_p1_B = structure_loss(p1_B, 1 - gts)

                loss = loss_pred  + 0.01*T_Loss + loss_p1_B

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_gradient(optimizer, opt.clip)
            scaler.step(optimizer)
            scaler.update()

            step += 1
            epoch_step += 1
            loss_all += loss.data

            if i % 20 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} '.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data))
                logging.info(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} '.
                    format(epoch, opt.epoch, i, total_step, loss.data))

        loss_all /= epoch_step
        logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if epoch % 50 == 0:
            torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch + 1))
        print('Save checkpoints successfully!')
        raise


def val(test_loader, model, epoch, save_path, writer):
    """
    validation function
    """
    if epoch < 10:
        return

    # 全局变量
    global best_metric_dict, best_score_mae, best_epoch, best_score
    global global_best_score, global_best_metric_dict, global_best_epoch
    global local_best_score, local_best_metric_dict, local_best_epoch

    FM = Measure.Fmeasure()
    SM = Measure.Smeasure()
    EM = Measure.Emeasure()
    WFM = Measure.WeightedFmeasure()
    MAE = Measure.MAE()
    metrics_dict = dict()

    model.eval()
    test_loss = 0
    step = 0
    with torch.no_grad(),autocast():
        for i in range(test_loader.size):
            image, gt, gt_tensor, _, _ = test_loader.load_data()
            gts = gt_tensor.unsqueeze(1).cuda()
            gt = np.asarray(gt, np.float32)
            image = image.cuda()

            res, pcnn,_,_ = model(image)

            loss_pred = structure_loss(res, gts)
            test_loss += loss_pred.data
            step += 1

            res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            FM.step(pred=res, gt=gt)
            SM.step(pred=res, gt=gt)
            EM.step(pred=res, gt=gt)
            WFM.step(pred=res, gt=gt)
            MAE.step(pred=res, gt=gt)

        test_loss /= step
        writer.add_scalar('test_pred_loss', test_loss, global_step=epoch)

        metrics_dict.update(Sm=SM.get_results()['sm'])
        metrics_dict.update(meanEm=EM.get_results()['em']['curve'].mean().round(3))
        metrics_dict.update(wFm=WFM.get_results()['wfm'].round(3))
        metrics_dict.update(mae=MAE.get_results()['mae'].round(3))

        cur_score_mae = metrics_dict['mae']
        cur_score = metrics_dict['Sm'] + metrics_dict['meanEm'] + metrics_dict['wFm']


        if epoch == 10:
            global_best_score = cur_score
            global_best_metric_dict = metrics_dict
            global_best_epoch = epoch
            torch.save(model.state_dict(), save_path + 'Net_epoch_best_overall.pth')
            print(f'>>> [Global Init] Save best model at epoch {epoch}')
        else:
            if cur_score > global_best_score:
                global_best_score = cur_score
                global_best_metric_dict = metrics_dict
                global_best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'Net_epoch_best_overall.pth')
                print(f'>>> [Global Update] Save new best model at epoch {epoch}')
            else:
                print(f'>>> [Global] No update, current best is epoch {global_best_epoch}')


        if (epoch - 1) % 5 == 0:
            local_best_score = -1
            local_best_metric_dict = None
            local_best_epoch = -1


        if cur_score > local_best_score:
            local_best_score = cur_score
            local_best_metric_dict = metrics_dict
            local_best_epoch = epoch
            window_start = epoch // 5 * 5
            window_end = (epoch // 5 + 1) * 5 - 1
            torch.save(model.state_dict(), save_path + f'Net_epoch_best_{window_start}_to_{window_end}.pth')
            print(f'>>> [Local Update] Save best in window {window_start}-{window_end}, epoch {epoch}')
        else:
            print('>>> [Local] Not best in this 5-epoch window.')

        # 打印日志
        print('[Cur Epoch: {}] Metrics ( Sm={}, wFm={}, meanEm={}, MAE={})'.format(
            epoch, metrics_dict['Sm'], metrics_dict['wFm'], metrics_dict['meanEm'], metrics_dict['mae']))
        logging.info('[Cur Epoch: {}] Metrics ( Sm={}, wFm={}, meanEm={}, MAE={})'.format(
            epoch, metrics_dict['Sm'], metrics_dict['wFm'], metrics_dict['meanEm'], metrics_dict['mae']))

        print('[Local Best Epoch: {}] Metrics ( Sm={}, wFm={}, meanEm={}, MAE={})'.format(
            local_best_epoch, local_best_metric_dict['Sm'], local_best_metric_dict['wFm'],
            local_best_metric_dict['meanEm'], local_best_metric_dict['mae']))

        print('[Global Best Epoch: {}] Metrics ( Sm={}, wFm={}, meanEm={}, MAE={})'.format(
            global_best_epoch, global_best_metric_dict['Sm'], global_best_metric_dict['wFm'],
            global_best_metric_dict['meanEm'], global_best_metric_dict['mae']))

        # TensorBoard
        writer.add_scalar('Test_Sm', metrics_dict['Sm'], global_step=epoch)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=75, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=14, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=75, help='every n epochs decay learning rate')#50
    parser.add_argument('--model', type=str, default='MyNet-P2T-large')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--train_root', type=str, default='./Dataset/TrainDataset/',#'./Dataset/SOD_Train/DUTS_TR/'  './Dataset/TrainDataset/'
                        help='the training rgb images root')
    parser.add_argument('--val_root', type=str, default='./Dataset/TestDataset/COD10K/',#'./Dataset/SOD_Test/DUTS-TE/'  './Dataset/TestDataset/COD10K/'
                        help='the test rgb images root')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='train use gpu')
    parser.add_argument('--save_path', type=str, default='./Checkpoints/MVGNet_v10_test0504/',
                        help='the path to save model and log')
    opt = parser.parse_args()

    # set the device for training
    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')
    cudnn.benchmark = True

    # build the model
    if opt.model == 'MyNet-PVTv2-B4':
        model = Network(channel=64, arc='PVTv2-B4', M=[8, 8, 8], N=[4, 8, 16]).cuda()
    elif opt.model == 'MyNet-PVTv2-B0':
        model = Network(channel=32, arc='PVTv2-B0', M=[8, 8, 8], N=[8, 16, 32]).cuda()
    elif opt.model == 'MyNet-PVTv2-B1':
        model = Network(channel=64, arc='PVTv2-B1', M=[8, 8, 8], N=[4, 8, 16]).cuda()
    elif opt.model == 'MyNet-PVTv2-B2':
        model = Network(channel=64, arc='PVTv2-B2', M=[8, 8, 8], N=[4, 8, 16]).cuda()
    elif opt.model == 'MyNet-PVTv2-B2-li':
        model = Network(channel=64, arc='PVTv2-B2-li', M=[8, 8, 8], N=[4, 8, 16]).cuda()
    elif opt.model == 'MyNet-PVTv2-B5':
        model = Network(channel=64, arc='PVTv2-B5', M=[8, 8, 8], N=[4, 8, 16]).cuda()

    elif opt.model == 'MyNet-P2T-base':
        model = Network(channel=64, arc='P2T-base').cuda()
    elif opt.model == 'MyNet-P2T-small':
        model = Network(channel=64, arc='P2T-small').cuda()
    elif opt.model == 'MyNet-P2T-tiny':
        model = Network(channel=48, arc='P2T-tiny').cuda()
    elif opt.model == 'MyNet-P2T-large':
        model = Network(channel=64, arc='P2T-large').cuda()
    else:
        raise Exception("Invalid Model Symbol: {}".format(opt.model))


    grad_loss_func = torch.nn.MSELoss()


    if opt.load is not None:

        model.load_state_dict(torch.load(opt.load))
        print('load model from ', opt.load)

    optimizer = torch.optim.Adam(model.parameters(), opt.lr,eps=1e-4)

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load data
    print('load data...')
    train_loader = get_loader(image_root=opt.train_root + 'Imgs/',
                              gt_root=opt.train_root + 'GT/',
                              batchsize=opt.batchsize,
                              trainsize=opt.trainsize,
                              num_workers=4)
    val_loader = test_dataset(image_root=opt.val_root + 'Imgs/',
                              gt_root=opt.val_root + 'GT/',
                              testsize=opt.trainsize)
    total_step = len(train_loader)

    # logging
    logging.basicConfig(filename=save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info(">>> current mode: network-train/val")
    logging.info('>>> config: {}'.format(opt))
    print('>>> config: : {}'.format(opt))

    step = 0
    writer = SummaryWriter(save_path + 'summary')

    best_score = 0
    best_epoch = 0

    best_metric_dict = {}
    best_score_mae = 0
    best_epoch = 0
    best_score = 0

    # 新增的全局变量
    global_best_score = 0
    global_best_metric_dict = {}
    global_best_epoch = 0
    local_best_score = -1
    local_best_metric_dict = None
    local_best_epoch = -1

    cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=1e-5)
    print(">>> start train...")

    scaler = GradScaler(init_scale=1024.0)

    for epoch in range(1, opt.epoch):
        # schedule
        cosine_schedule.step()
        writer.add_scalar('learning_rate', cosine_schedule.get_lr()[0], global_step=epoch)
        logging.info('>>> current lr: {}'.format(cosine_schedule.get_lr()[0]))
        # train

        train(train_loader, model, optimizer, epoch, save_path, writer,scaler)

        val(val_loader, model, epoch, save_path, writer)
