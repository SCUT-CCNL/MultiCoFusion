import random
from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import RandomSampler

from data_loaders import PathgraphomicDatasetLoader
from networks import define_net, define_reg, define_optimizer, define_scheduler, model_loss_layer
from utils import unfreeze_unimodal, CoxLoss, CIndex_lifeline, cox_log_rank, accuracy_cox, mixed_collate, count_parameters

import pdb
import pickle
import os
import time

def train(opt, data, device, k, graph):
    cudnn.deterministic = True
    torch.cuda.manual_seed_all(2019)
    torch.manual_seed(2019)
    random.seed(2019)

    model = define_net(opt, k)
    model_loss = model_loss_layer().to(device) ### For HU and RHU, not called in this version
    optimizer = define_optimizer(opt, model)
    scheduler = define_scheduler(opt, optimizer)
    print(model)
    print("Number of Trainable Parameters: %d" % count_parameters(model))
    print("Activation Type For Surv:", opt.act_type_1)
    print("Activation Type For Grade:", opt.act_type_2)
    print("Optimizer Type:", opt.optimizer_type)
    print("Regularization Type:", opt.reg_type)

    use_patch, roi_dir = '_', 'all_st'

    custom_data_loader = PathgraphomicDatasetLoader(opt, data, split='train', mode=opt.mode)
    train_loader = torch.utils.data.DataLoader(dataset=custom_data_loader, batch_size=opt.batch_size, shuffle=True, collate_fn=mixed_collate, num_workers=5)
    metric_logger = {'train':{'loss':[], 'pvalue':[], 'cindex':[], 'surv_acc':[], 'grad_acc':[]},
                      'test':{'loss':[], 'pvalue':[], 'cindex':[], 'surv_acc':[], 'grad_acc':[]}}

    for epoch in tqdm(range(opt.epoch_count, opt.niter+opt.niter_decay+1)):
        if opt.finetune == 1:
            unfreeze_unimodal(opt, model, epoch)

        model.train()
        risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])    # Used for calculating the C-Index
        loss_epoch, grad_acc_epoch = 0, 0

        for batch_idx, (x_path, x_grph, x_omic, censor, survtime, grade, clin, index) in enumerate(train_loader):#
            FLAG = batch_idx%2
            censor = censor.to(device) if "surv" in opt.task else censor
            grade = grade.to(device) if "grad" in opt.task else grade
            x_path = x_path.to(device)
            x_grph = x_grph.to(device)
            x_omic = x_omic.to(device)
            clin = clin.to(device) ### Not used in this version

            loss_reg = define_reg(opt, model)
            if FLAG:
                _, _, pred_surv, _ = model(x_path=x_path, x_grph=x_grph, x_omic=x_omic, gene_adj=graph, clin=clin)
                loss_nll = 0
                loss_cox = CoxLoss(survtime, censor, pred_surv, device)
            else:
                _, _, _, pred_grad = model(x_path=x_path, x_grph=x_grph, x_omic=x_omic, gene_adj=graph, clin=clin)
                loss_nll = F.nll_loss(pred_grad, grade)
                loss_cox = 0
            loss = opt.lambda_cox*loss_cox + opt.lambda_nll*loss_nll + opt.lambda_reg*loss_reg

            # ### DWA
            # if index_epoch == 0 or index_epoch == 1:
            #     lambda_weight[:, index_epoch] = 1.0
            # else:
            #     w_1 = avg_cost[index_epoch - 1, 0] / avg_cost[index_epoch - 2, 0]
            #     w_2 = avg_cost[index_epoch - 1, 1] / avg_cost[index_epoch - 2, 1]
            #     w_3 = avg_cost[index_epoch - 1, 2] / avg_cost[index_epoch - 2, 2]
            #     lambda_weight[0, index_epoch] = 3 * np.exp(w_1 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))
            #     lambda_weight[1, index_epoch] = 3 * np.exp(w_2 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))
            #     lambda_weight[2, index_epoch] = 3 * np.exp(w_3 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))
            #     # lambda_weight[0, index_epoch] = 2 * np.exp(w_1 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T))
            #     # lambda_weight[1, index_epoch] = 2 * np.exp(w_2 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T))
            # avg_cost[index_epoch,0]+=loss_cox/len(train_loader)
            # avg_cost[index_epoch,1]+=loss_nll/len(train_loader)
            # avg_cost[index_epoch,2]+=loss_reg/len(train_loader)
            # loss = lambda_weight[0, index_epoch] * loss_cox + lambda_weight[1, index_epoch] * loss_nll + lambda_weight[2, index_epoch] * loss_reg
            # # loss = lambda_weight[0, index_epoch] * loss_cox + lambda_weight[1, index_epoch] * loss_nll
            # loss = loss + loss_reg
            # ### END of DWA

            ### HU and RHU
            # loss = model_loss(loss_cox, loss_nll, loss_reg)
            # loss = loss+loss_reg
            ###

            loss_epoch += loss.data.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if opt.task == "surv" or FLAG:
                risk_pred_all = np.concatenate((risk_pred_all, pred_surv.detach().cpu().numpy().reshape(-1)))
                censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))
                survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))
            elif opt.task == "grad" or not FLAG:
                pred_grad = pred_grad.argmax(dim=1, keepdim=True)
                grad_acc_epoch += pred_grad.eq(grade.view_as(pred_grad)).sum().item()

            if opt.verbose > 0 and opt.print_every > 0 and (batch_idx % opt.print_every == 0 or batch_idx+1 == len(train_loader)):
                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                    epoch+1, opt.niter+opt.niter_decay, batch_idx+1, len(train_loader), loss.item()))
        scheduler.step()

        if opt.measure or epoch == (opt.niter+opt.niter_decay - 1):
            loss_epoch /= len(train_loader.dataset)

            cindex_epoch = CIndex_lifeline(risk_pred_all, censor_all, survtime_all) if opt.task == 'surv' or opt.task == 'surv_grad' else None
            pvalue_epoch = cox_log_rank(risk_pred_all, censor_all, survtime_all)  if opt.task == 'surv' or opt.task == 'surv_grad' else None
            surv_acc_epoch = accuracy_cox(risk_pred_all, censor_all)  if opt.task == 'surv' or opt.task == 'surv_grad' else None
            grad_acc_epoch = grad_acc_epoch / (len(train_loader.dataset)/2) if opt.task == 'grad' or opt.task == 'surv_grad' else None
            loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test = test(opt, model, data, 'test', device, graph, model_loss)

            metric_logger['train']['loss'].append(loss_epoch)
            metric_logger['train']['cindex'].append(cindex_epoch)
            metric_logger['train']['pvalue'].append(pvalue_epoch)
            metric_logger['train']['surv_acc'].append(surv_acc_epoch)
            metric_logger['train']['grad_acc'].append(grad_acc_epoch)

            metric_logger['test']['loss'].append(loss_test)
            metric_logger['test']['cindex'].append(cindex_test)
            metric_logger['test']['pvalue'].append(pvalue_test)
            metric_logger['test']['surv_acc'].append(surv_acc_test)
            metric_logger['test']['grad_acc'].append(grad_acc_test)

            pickle.dump(pred_test, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%d%s%d_pred_test.pkl' % (opt.model_name, k, use_patch, epoch)), 'wb'))
            print()
            if opt.verbose > 0:
                if opt.task == 'surv':
                    print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}'.format('Train', loss_epoch, 'C-Index', cindex_epoch))
                    print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}\n'.format('Test', loss_test, 'C-Index', cindex_test))
                elif opt.task == 'grad':
                    print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}'.format('Train', loss_epoch, 'Accuracy', grad_acc_epoch))
                    print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}\n'.format('Test', loss_test, 'Accuracy', grad_acc_test))
                elif opt.task == "surv_grad":
                    print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}, {:s}: {:.4f}'.format('Train', loss_epoch, 'C-Index', cindex_epoch, 'Accuracy', grad_acc_epoch))
                    print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}, {:s}: {:.4f}\n'.format('Test', loss_test, 'C-Index', cindex_test, 'Accuracy', grad_acc_test))

            # if opt.task == 'grad' and loss_epoch < opt.patience:
            #     print("Early stopping at Epoch %d" % epoch)
            #     break

    return model, optimizer, metric_logger, model_loss


def test(opt, model, data, split, device, graph, model_loss):
    model.eval()

    custom_data_loader = PathgraphomicDatasetLoader(opt, data, split=split, mode=opt.mode)
    test_loader = torch.utils.data.DataLoader(dataset=custom_data_loader, batch_size=opt.batch_size, shuffle=False, collate_fn=mixed_collate, num_workers=5)
    
    risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])
    probs_all, gt_all = None, np.array([])
    loss_test, grad_acc_test = 0, 0


    for batch_idx, (x_path, x_grph, x_omic, censor, survtime, grade, clin, index) in enumerate(test_loader): #index add by KT
        censor = censor.to(device) if "surv" in opt.task else censor
        grade = grade.to(device) if "grad" in opt.task else grade
        if opt.task == "surv_grad":
            _, _, pred_surv, pred_grad = model(x_path=x_path.to(device), x_grph=x_grph.to(device), x_omic=x_omic.to(device), gene_adj=graph.to(device), clin = clin)
        elif opt.task == "grad":
            _, _, _, pred_grad = model(x_path=x_path.to(device), x_grph=x_grph.to(device), x_omic=x_omic.to(device), gene_adj=graph.to(device), clin = clin)
        elif opt.task == "surv":
            _, _, pred_surv, _ = model(x_path=x_path.to(device), x_grph=x_grph.to(device), x_omic=x_omic.to(device), gene_adj=graph.to(device), clin = clin)


        loss_cox = CoxLoss(survtime, censor, pred_surv, device) if opt.task == "surv" or opt.task == "surv_grad" else 0
        loss_reg = define_reg(opt, model)
        loss_nll = F.nll_loss(pred_grad, grade) if opt.task == "grad" or opt.task == "surv_grad" else 0

        loss = opt.lambda_cox * loss_cox + opt.lambda_nll * loss_nll + opt.lambda_reg * loss_reg
        
        loss_test += loss.data.item()

        gt_all = np.concatenate((gt_all, grade.detach().cpu().numpy().reshape(-1)))

        if opt.task == "surv":
            risk_pred_all = np.concatenate((risk_pred_all, pred_surv.detach().cpu().numpy().reshape(-1)))
            censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))
            survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))
        elif opt.task == "grad":
            grade_pred = pred_grad.argmax(dim=1, keepdim=True)
            grad_acc_test += grade_pred.eq(grade.view_as(grade_pred)).sum().item()
            probs_np = pred_grad.detach().cpu().numpy()
            probs_all = probs_np if probs_all is None else np.concatenate((probs_all, probs_np), axis=0)
        elif opt.task == "surv_grad":
            risk_pred_all = np.concatenate((risk_pred_all, pred_surv.detach().cpu().numpy().reshape(-1)))
            censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))
            survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))
            grade_pred = pred_grad.argmax(dim=1, keepdim=True)
            grad_acc_test += grade_pred.eq(grade.view_as(grade_pred)).sum().item()
            probs_np = pred_grad.detach().cpu().numpy()
            probs_all = probs_np if probs_all is None else np.concatenate((probs_all, probs_np), axis=0)

    ################################################### 
    # ==== Measuring Test Loss, C-Index, P-Value ==== #
    ###################################################
    loss_test /= len(test_loader)
    cindex_test = CIndex_lifeline(risk_pred_all, censor_all, survtime_all) if opt.task == 'surv' or opt.task == 'surv_grad' else None
    pvalue_test = cox_log_rank(risk_pred_all, censor_all, survtime_all) if opt.task == 'surv' or opt.task == 'surv_grad' else None
    surv_acc_test = accuracy_cox(risk_pred_all, censor_all) if opt.task == 'surv' or opt.task == 'surv_grad' else None
    grad_acc_test = grad_acc_test / len(test_loader.dataset) if opt.task == 'grad' or opt.task == 'surv_grad' else None
    pred_test = [risk_pred_all, survtime_all, censor_all, probs_all, gt_all]

    return loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test