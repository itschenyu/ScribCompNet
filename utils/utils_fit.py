import os
import random
import torch.nn as nn
import torch
import torch.nn.functional as F
from nets.ScribCompNet_training import CE_Loss, Dice_loss, Focal_Loss, pDLoss, SCEloss
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import f_score


loss_sce_kernels_desc_defaults = [{"weight": 1, "xy": 6, "rgb": 0.1}]
loss_sce_radius = 5
z = 1
l = 0.1
m = 0.1
n = 0.25

def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(3, 1, 1)(x)
    mu_y = nn.AvgPool2d(3, 1, 1)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(3, 1, 1)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(3, 1, 1)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(3, 1, 1)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d

    return torch.clamp((1 - SSIM) / 2, 0, 1)


def ScaleAdaptiveHarmony(x, y, alpha):
    ssim = torch.mean(SSIM(x,y))
    l1_loss = torch.mean(torch.abs(x-y))
    loss_sah = alpha*ssim + (1-alpha)*l1_loss
    return loss_sah

def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss      = 0
    total_f_score   = 0

    val_loss        = 0
    val_f_score     = 0

    pD_loss = pDLoss(num_classes, ignore_index=num_classes).cuda()
    sce = SCEloss().cuda()

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step: 
            break
        imgs, pngs, labels, srbs, srb_labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                srbs    = srbs.cuda(local_rank)
                srb_labels = srb_labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:
            outputs = model_train(imgs)
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
            else:
                loss_ce1 = CE_Loss(outputs[0], srbs, weights, num_classes = num_classes)
                loss_ce2 = CE_Loss(outputs[1], srbs, weights, num_classes = num_classes)
                loss_ce = 0.5 * (loss_ce1 + loss_ce2)

            beta = random.random() + 1e-10
            outputs_soft = [torch.softmax(output, dim=1) for output in outputs]
            weighted_sum = beta * outputs_soft[0].detach() + (1.0-beta) * outputs_soft[1].detach()
            pseudo_supervision = torch.argmax(weighted_sum, dim=1, keepdim=False)
            losses_pse_sup = [pD_loss(outputs_soft[i], pseudo_supervision.unsqueeze(1)) for i in range(2)]
            loss_pse_sup = sum(losses_pse_sup) / len(losses_pse_sup)

            image_scale = F.interpolate(imgs, scale_factor=0.75, mode='bilinear', align_corners=True, recompute_scale_factor=True)
            outputs_t = model_train(image_scale)
            out2_scale = F.interpolate(outputs[0], scale_factor=0.75, mode='bilinear', align_corners=True, recompute_scale_factor=True)
            loss_sah = ScaleAdaptiveHarmony(outputs_t[0], out2_scale, 0.85)

            image_ = F.interpolate(imgs, scale_factor=0.5, mode='bilinear', align_corners=True, recompute_scale_factor=True)
            sample = {'rgb': image_}
            out_ = F.interpolate(outputs_soft[0], scale_factor=0.5, mode='bilinear', align_corners=True, recompute_scale_factor=True)
            loss_sce = sce(out_, loss_sce_kernels_desc_defaults, loss_sce_radius, sample, image_.shape[2], image_.shape[3])['loss']

            loss = z * loss_ce + m * loss_pse_sup + l * loss_sce + n * loss_sah

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss      = loss + main_dice

            with torch.no_grad():
                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model_train(imgs)
                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
                else:
                    loss_ce1 = CE_Loss(outputs[0], srbs, weights, num_classes = num_classes)
                    loss_ce2 = CE_Loss(outputs[1], srbs, weights, num_classes = num_classes)
                    loss_ce = 0.5 * (loss_ce1 + loss_ce2)
                
                beta = random.random() + 1e-10
                outputs_soft = [torch.softmax(output, dim=1) for output in outputs]
                weighted_sum = beta * outputs_soft[0].detach() + (1.0-beta) * outputs_soft[1].detach()
                pseudo_supervision = torch.argmax(weighted_sum, dim=1, keepdim=False)
                losses_pse_sup = [pD_loss(outputs_soft[i], pseudo_supervision.unsqueeze(1)) for i in range(2)]
                loss_pse_sup = sum(losses_pse_sup) / len(losses_pse_sup)

                image_scale = F.interpolate(imgs, scale_factor=0.75, mode='bilinear', align_corners=True, recompute_scale_factor=True)
                outputs_t = model_train(image_scale)
                out2_scale = F.interpolate(outputs[0], scale_factor=0.75, mode='bilinear', align_corners=True, recompute_scale_factor=True)
                loss_sah = ScaleAdaptiveHarmony(outputs_t[0], out2_scale, 0.85)

                image_ = F.interpolate(imgs, scale_factor=0.5, mode='bilinear', align_corners=True, recompute_scale_factor=True)
                sample = {'rgb': image_}
                out_ = F.interpolate(outputs_soft[0], scale_factor=0.5, mode='bilinear', align_corners=True, recompute_scale_factor=True)
                loss_sce = sce(out_, loss_sce_kernels_desc_defaults, loss_sce_radius, sample, image_.shape[2], image_.shape[3])['loss']

                loss = z * loss_ce + m * loss_pse_sup + l * loss_sce + n * loss_sah

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss      = loss + main_dice

                with torch.no_grad():
                    _f_score = f_score(outputs[0], labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss      += loss.item()
        total_f_score   += _f_score.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'f_score'   : total_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, labels, srbs, srb_labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                srbs    = srbs.cuda(local_rank)
                srb_labels = srb_labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

            outputs = model_train(imgs)
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
            else:
                loss_ce1 = CE_Loss(outputs[0], srbs, weights, num_classes = num_classes)
                loss_ce2 = CE_Loss(outputs[1], srbs, weights, num_classes = num_classes)
                loss_ce = 0.5 * (loss_ce1 + loss_ce2)

            outputs_soft = [torch.softmax(output, dim=1) for output in outputs]
            weighted_sum = beta * outputs_soft[0].detach() + (1.0-beta) * outputs_soft[1].detach()
            pseudo_supervision = torch.argmax(weighted_sum, dim=1, keepdim=False)
            losses_pse_sup = [pD_loss(outputs_soft[i], pseudo_supervision.unsqueeze(1)) for i in range(2)]
            loss_pse_sup = sum(losses_pse_sup) / len(losses_pse_sup)

            image_scale = F.interpolate(imgs, scale_factor=0.75, mode='bilinear', align_corners=True, recompute_scale_factor=True)
            outputs_t = model_train(image_scale)
            out2_scale = F.interpolate(outputs[0], scale_factor=0.75, mode='bilinear', align_corners=True, recompute_scale_factor=True)
            loss_sah = ScaleAdaptiveHarmony(outputs_t[0], out2_scale, 0.85)

            image_ = F.interpolate(imgs, scale_factor=0.5, mode='bilinear', align_corners=True, recompute_scale_factor=True)
            sample = {'rgb': image_}
            out_ = F.interpolate(outputs_soft[0], scale_factor=0.5, mode='bilinear', align_corners=True, recompute_scale_factor=True)
            loss_sce = sce(out_, loss_sce_kernels_desc_defaults, loss_sce_radius, sample, image_.shape[2], image_.shape[3])['loss']
            
            loss = z * loss_ce + m * loss_pse_sup + l * loss_sce + n * loss_sah

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss  = loss + main_dice
            _f_score    = f_score(outputs[0], labels)

            val_loss    += loss.item()
            val_f_score += _f_score.item()
            
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1),
                                'f_score'   : val_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
            
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss/ epoch_step, val_loss/ epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth'%((epoch + 1), total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            if (epoch + 1) >= (Epoch / 2):
                torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights_%d.pth"%(epoch + 1)))
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))