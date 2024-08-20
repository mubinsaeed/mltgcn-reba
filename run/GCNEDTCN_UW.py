import torch.optim as optim
import random
from losses.loss_UW import *
from models.model_MT_UW import *
from util.Uwdatareader_UW import *
#from val.validate_model_UW import val, EarlyStopping  Todo
#from vis.plotCM import * Todo
import sklearn
import math
from config_files.config_UW import *
from tensorboardX import SummaryWriter
#import torchinfo


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def _init_fn(worker_id):
    np.random.seed(int(seed))


# %% Training
base_data_dir = config_data['base_data_dir']
#train_split = np.load(base_data_dir + config_data['train_dir'])
#val_split = np.load(base_data_dir + config_data['val_dir'])
train_split = np.array(['01','05','04','10','03','06','08','09','11'])
val_split = np.array(['02','07'])
def train(generator_train, generator_val, model_, mt_losses, optimizer_, lr_):
    global vallepochloss
    writer = SummaryWriter(f"testing/lr {lr_}/data")
    output_file = open(config_exp['log_dir'] + config_exp['output_name'], 'a')
    output_file.write('\n------------------------------------------------------------------------\n')
    output_file.write(what_is_different_in_this_code)
    output_file.write('\n------------------------------------------------------------------------\n')
    output_file.write('\n------- lr: ' + str(lr_) + ', batch_size: ' + str(config_exp['BATCH_SIZE']) + '-----------\n')
    output_file.close()
    output_file = open(config_exp['log_dir'] + 'maps_' + config_exp['output_name'], 'a')
    output_file.write('\n------------------------------------------------------------------------\n')
    output_file.write(what_is_different_in_this_code)
    output_file.write('\n------------------------------------------------------------------------\n')
    output_file.write('\n------- lr: ' + str(lr_) + ', batch_size: ' + str(config_exp['BATCH_SIZE']) + '-----------\n')
    output_file.close()
    # initialize the early_stopping object
    #early_stopping = EarlyStopping(patience=config_exp['PATIENCE'], verbose=True)

    for epoch in range(config_exp['STEPS']):
        output_file = open(config_exp['log_dir'] + config_exp['output_name'], 'a')
        losses = 0.0
        losses_class = 0.0
        losses_reg = 0.0
        for local_im, reba_gt in generator_train:
            local_im,  reba_gt = local_im.float().cuda(), reba_gt.float().cuda()
            #local_im = np.array()
            loss_class, loss_reg, loss = mt_losses(local_im, [reba_gt])
            optimizer_.zero_grad()
            loss.backward()
            optimizer_.step()
            for p in mt_losses.eta:
                p.data.clamp_(0.5)  #for balancing the both loss function values so one doesn't have larger impact on total_loss
            losses = losses + loss.cpu().data.numpy()
            losses_class = losses_class + loss_class.cpu().data.numpy()
            losses_reg = losses_reg + loss_reg.cpu().data.numpy()
            #writer.add_scalar('Training/Batch Loss Reg', losses_reg.item(), global_step=stepp)
           # stepp += 1
            #writer.flush()

        print(epoch, ': Train: ', np.round(losses, 4), ' Train_class: ',
              np.round(losses_class, 4), ' Train_reg: ', np.round(losses_reg, 4))
        writer.add_scalar('Training/Epoch Loss Reg', losses_reg.item(), global_step=epoch)

        vallosses, vallosses_reg, vallosses_class = val(generator_val, model_, mt_losses)
        writer.add_scalar('Validation/Epoch Loss Reg', vallosses_reg.item(), global_step=epoch)

        print(epoch, ': Train: ', np.round(losses, 4), ' Val: ', np.round(vallosses, 4), ' Train_class: ',
              np.round(losses_class, 4), ' Train_reg: ', np.round(losses_reg, 4), ' Val_class: ', np.round(vallosses_class, 4),
              ' Val_reg: ', np.round(vallosses_reg, 4))
        #writer.add_hparams({'lr': lr_}, {'Trainingloss':losses_reg.item(),'Validationloss': vallosses_reg.item()},global_step=epoch)
        writer.flush()

        output_file.write(
            'EPOCH: %02d\t TrainLoss: %0.04f \t ValLoss: %0.04f \t Train_class: %0.04f \t Train_reg: %0.04f \t Val_class: %0.04f \t Val_reg: %0.04f\n' % (
                epoch, np.round(losses, 4), np.round(vallosses, 4), np.round(losses_class, 4), np.round(losses_reg, 4),
                np.round(vallosses_class, 4), np.round(vallosses_reg, 4)))

        output_file.close()
        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        if math.isnan(vallosses):
            break
        else:
            early_stopping(val_loss=vallosses, model=model_, mt_losses=mt_losses, dir_out=config_exp['checkpoint_dir'],
                           outputfile=config_exp['output_name'], CM_dir=config_exp['CM_dir'], lr=lr_,
                           epoch=epoch, generator_val=generator_val, temporal_len=max_len,
                           n_classes=config_exp['NUMBER_OF_CLASSES'])
        if early_stopping.early_stop:
            print("Early stopping")
            break
    writer.close()

# %% Data preprocessing
print(colored('---------------------------- Pre-processing ----------------------------', 'green'))
print(colored('batch_size: ' + str(config_exp['BATCH_SIZE']), 'green'))
params_train = {'batch_size': config_exp['BATCH_SIZE'], 'shuffle': True}
training_set = Dataset_with_REBA(train_split, history=HISTORY)
training_generator = data.DataLoader(training_set, num_workers=0, pin_memory=True, worker_init_fn=_init_fn,
                                     **params_train)
params_val = {'batch_size': config_exp['BATCH_SIZE'], 'shuffle': False}
val_set = Dataset_with_REBA(val_split, history=HISTORY)  # Dataset_with_REBA
val_generator = data.DataLoader(val_set, num_workers=0, pin_memory=True, worker_init_fn=_init_fn, **params_val)
n_layers = len(config_exp['n_nodes'])
max_len = max(np.max(training_set.max_len), np.max(val_set.max_len))
max_len = int(np.ceil(max_len / (2 ** n_layers))) * 2 ** n_layers
training_set.mask_data(max_len, mask_value=-1)
val_set.mask_data(max_len, mask_value=-1)
print(colored('Maximum sequence length: ' + str(max_len), 'blue'))
print(colored('Size of input (training set): ' + str((
    len(training_set.poselist), training_set.poselist[-1].shape[0],
    training_set.poselist[-1].shape[1],
    training_set.poselist[-1].shape[2])), 'blue'))
#print(colored(
#    'Size of output labels (training set): ' + str((len(training_set.labellist), len(training_set.labellist[-1]))),
#    'blue'))
print(colored('Size of output reba scores (training set): ' + str(
    (len(training_set.rebascorelist), training_set.rebascorelist[-1].shape[0])), 'blue'))

print(colored('Size of input (validation set): ' + str((
    len(val_set.poselist), val_set.poselist[-1].shape[0],
    val_set.poselist[-1].shape[1],
    val_set.poselist[-1].shape[2])), 'blue'))
#print(colored(
#    'Size of output labels (validation set): ' + str((len(val_set.labellist), len(val_set.labellist[-1]))),
#    'blue'))
print(colored('Size of output reba scores (validation set): ' + str(
    (len(val_set.rebascorelist), val_set.rebascorelist[-1].shape[0])), 'blue'))

# %% Training
print(colored('---------------------------- Training ----------------------------', 'green'))
for lr in config_exp['LR']:

    print('---------------------------- lr: ', lr, '----------------------------')
    vallepochloss = np.inf
    if config_exp['TASK'] == 'MTL-Emb':
        model = gcnEdtcnREBA_emb(hidden=config_exp['n_nodes'], kernel_size=4)
    elif config_exp['TASK'] == 'MTL':
        model = gcnEdtcnREBA_tanh(hidden=config_exp['n_nodes'], kernel_size=4)
    elif config_exp['TASK'] == 'classification':
        model = gcnEdtcn_class(hidden=config_exp['n_nodes'], kernel_size=4)
    elif config_exp['TASK'] == 'regression':
        model = gcn_reg(hidden=config_exp['n_nodes'], kernel_size=4)

    model.cuda()
    model.train()
    model.apply(weightinit)

    if config_exp['loss_reg'] == 'MSE' or config_exp['loss_reg'] == 'L1' or config_exp['loss_reg'] == 'SmoothL1':
        #MT_losses = RegLoss(model=model, loss_fn=criterion_class + criterion_reg).cuda()
        MT_losses = RegLoss(model=model, loss_fn=  criterion_reg).cuda()
    elif config_exp['loss_reg'] == 'MSEL1' or config_exp['loss_reg'] == 'MSESmoothL1':
        MT_losses = MultiTask3Loss(model=model, loss_fn=criterion_class + criterion_reg).cuda()

    optimizer = optim.Adam(MT_losses.parameters(), lr=float(lr))


#torchinfo.summary(model,input_size=(1,1,27,3)) #batchsize,noofsamples,value,xyz
    train(training_generator, val_generator, model, MT_losses, optimizer, float(lr))


