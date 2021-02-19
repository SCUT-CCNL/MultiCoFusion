import logging
import pickle

from data_loaders import *
from options import *
from train_test import train, test

from utils import getAggHazardCV, calcGradMetrics, CI_pm


### 1. Initializes parser and device
opt = my_parse_args_pathomic()
device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
print("Using device:", device)
if not os.path.exists(opt.checkpoints_dir): os.makedirs(opt.checkpoints_dir)
if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.exp_name)): os.makedirs(os.path.join(opt.checkpoints_dir, opt.exp_name))
if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name)): os.makedirs(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name))

### 2. Initializes Data
use_patch, roi_dir = '_', 'all_st'
data_cv_path = '%s/splits/gbmlgg15cv_all_st_1_1_0_rnaseq_new.pkl' % (opt.dataroot)
print("Loading %s" % data_cv_path)
print("Loading %s" % data_cv_path)
data_cv = pickle.load(open(data_cv_path, 'rb'))
data_cv_splits = data_cv['cv_splits']
graph = np.array(data_cv['graph'])
graph = torch.tensor(graph).type((torch.FloatTensor))
graph = graph.to(device)

results_surv = []
results_grad = []

### 3. Sets-Up Main Loop
for k, data in data_cv_splits.items(): ###data: censor(e), survtime(t), grade(g)
	print("*******************************************")
	print("************** SPLIT (%d/%d) **************" % (k, len(data_cv_splits.items())))
	print("*******************************************")
	### 3.1 Trains Model
	print("train: ", len(data_cv['cv_splits'][k]['train']['x_patname']))
	print("test: ", len(data_cv['cv_splits'][k]['test']['x_patname']))
	model, optimizer, metric_logger, model_loss = train(opt, data, device, k, graph)
	### 3.2 Evalutes Train + Test Error, and Saves Model
	loss_train, cindex_train, pvalue_train, surv_acc_train, grad_acc_train, pred_train = test(opt, model, data, 'train', device, graph, model_loss)
	loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test = test(opt, model, data, 'test', device, graph, model_loss)

	if opt.task == 'surv':
		print("[Final] Apply model to training set: C-Index: %.10f, P-Value: %.10e" % (cindex_train, pvalue_train))
		logging.info("[Final] Apply model to training set: C-Index: %.10f, P-Value: %.10e" % (cindex_train, pvalue_train))
		print("[Final] Apply model to testing set: C-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
		logging.info("[Final] Apply model to testing set: cC-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
		results_surv.append(cindex_test)
	elif opt.task == 'grad':
		print("[Final] Apply model to training set: Loss: %.10f, Acc: %.4f" % (loss_train, grad_acc_train))
		logging.info("[Final] Apply model to training set: Loss: %.10f, Acc: %.4f" % (loss_train, grad_acc_train))
		print("[Final] Apply model to testing set: Loss: %.10f, Acc: %.4f" % (loss_test, grad_acc_test))
		logging.info("[Final] Apply model to testing set: Loss: %.10f, Acc: %.4f" % (loss_test, grad_acc_test))
		results_grad.append(grad_acc_test)
	elif opt.task == "surv_grad":
		print("[Final] Apply model to training set: C-Index: %.10f, P-Value: %.10e" % (cindex_train, pvalue_train))
		logging.info("[Final] Apply model to training set: C-Index: %.10f, P-Value: %.10e" % (cindex_train, pvalue_train))
		print("[Final] Apply model to testing set: C-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
		logging.info("[Final] Apply model to testing set: cC-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
		results_surv.append(cindex_test)
		print("[Final] Apply model to training set: Loss: %.10f, Acc: %.4f" % (loss_train, grad_acc_train))
		logging.info("[Final] Apply model to training set: Loss: %.10f, Acc: %.4f" % (loss_train, grad_acc_train))
		print("[Final] Apply model to testing set: Loss: %.10f, Acc: %.4f" % (loss_test, grad_acc_test))
		logging.info("[Final] Apply model to testing set: Loss: %.10f, Acc: %.4f" % (loss_test, grad_acc_test))
		results_grad.append(grad_acc_test)

	### 3.3 Saves Model
	if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
		model_state_dict = model.module.cpu().state_dict()
	else:
		model_state_dict = model.cpu().state_dict()

	torch.save({
		'split':k,
		'opt': opt,
		'epoch': opt.niter+opt.niter_decay,
		'data': data,
		'model_state_dict': model_state_dict,
		'optimizer_state_dict': optimizer.state_dict(),
		'metrics': metric_logger},
		os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%d.pt' % (opt.model_name, k)))

	pickle.dump(pred_train, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%d%spred_train.pkl' % (opt.model_name, k, use_patch)), 'wb'))
	pickle.dump(pred_test, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%d%spred_test.pkl' % (opt.model_name, k, use_patch)), 'wb'))


print('Split Results:', results_surv)
print('Split Results:', results_grad)
print("Average:", np.array(results_surv).mean())
print("Average:", np.array(results_grad).mean())
pickle.dump(results_surv, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_results_surv.pkl' % opt.model_name), 'wb'))
pickle.dump(results_grad, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_results_grad.pkl' % opt.model_name), 'wb'))

models = [opt.model_name]
model_names = [opt.model_name]
if opt.task == "surv":
	ckpt_name = './checkpoints/TCGA_GBMLGG/' + opt.exp_name
	cv_surv = [np.array(getAggHazardCV(ckpt_name=ckpt_name, model=model)) for model in models]
	cv_surv = pd.DataFrame(np.array(cv_surv))
	cv_surv.columns = ['Split %s' % str(k) for k in range(1,16)]
	cv_surv.index = model_names
	cv_surv['C-Index'] = [CI_pm(cv_surv.loc[model]) for model in model_names]
	print("C-index: ", cv_surv[['C-Index']])
elif opt.task == "grad":
	ckpt_name = './checkpoints/TCGA_GBMLGG/' + opt.exp_name
	cv_grad = [calcGradMetrics(ckpt_name=ckpt_name, model=model, avg='micro') for model in models]
	cv_grad = pd.DataFrame(np.stack(cv_grad))
	cv_grad.columns = ['AUC', 'Avg Precision', 'F1-Score', 'F1-Score (Grade IV)']
	cv_grad.index = model_names
	print("'AUC', 'Avg Precision', 'F1-Score', 'F1-Score (Grade IV)'")
	for i in np.array(cv_grad)[0]:
		print(i, end=',')
elif opt.task == "surv_grad":
	ckpt_name = './checkpoints/TCGA_GBMLGG/' + opt.exp_name
	cv_surv = [np.array(getAggHazardCV(ckpt_name=ckpt_name, model=model)) for model in models]
	cv_surv = pd.DataFrame(np.array(cv_surv))
	cv_surv.columns = ['Split %s' % str(k) for k in range(1,16)]
	cv_surv.index = model_names
	cv_surv['C-Index'] = [CI_pm(cv_surv.loc[model]) for model in model_names]
	print("C-index: ", cv_surv[['C-Index']])
	ckpt_name = './checkpoints/TCGA_GBMLGG/' + opt.exp_name
	cv_grad = [calcGradMetrics(ckpt_name=ckpt_name, model=model, avg='micro') for model in models]
	cv_grad = pd.DataFrame(np.stack(cv_grad))
	cv_grad.columns = ['AUC', 'Avg Precision', 'F1-Score', 'F1-Score (Grade IV)']
	cv_grad.index = model_names
	print("'AUC', 'Avg Precision', 'F1-Score', 'F1-Score (Grade IV)'")
	for i in np.array(cv_grad)[0]:
		print(i, end=',')
