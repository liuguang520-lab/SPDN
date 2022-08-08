import numpy as np
import torch
import torch.optim
import os

from methods.backbone import model_dict
from data.datamgr import SimpleDataManager, SetDataManager

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 5-shot model use this
# from methods.SSL_5shot import GnnNet1

# 1-shot model use this
from methods.SSL_1shot import GnnNet1



from options import parse_args,get_best_file

def train1(base_loader, val_loader, model1, start_epoch, stop_epoch, params):

  # get optimizer and checkpoint path
  optimizer = torch.optim.Adam(model1.gnn.parameters())


  if not os.path.isdir(params.checkpoint_dir):
    os.makedirs(params.checkpoint_dir)

  # for validation
  global max_acc
  total_it = 0


  # start
  for epoch in range(start_epoch, stop_epoch):
    model1.train()

    total_it = model1.train_loop_refine(epoch, base_loader,  optimizer, total_it) #model are called by reference, no need to return

    model1.eval()

    acc = model1.test_loop_refine(val_loader)
    if acc > max_acc:
      print("best model! save...")
      max_acc = acc
      outfile = os.path.join(params.checkpoint_dir, 'best_model_s_q.tar')
      torch.save({'epoch':epoch, 'state':model1.state_dict()}, outfile)
    else:
      print("GG!  best accuracy_s_q {:f}".format(max_acc))

    if ((epoch + 1) % params.save_freq==0) or (epoch==stop_epoch-1):
      outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
      torch.save({'epoch':epoch, 'state':model1.state_dict()}, outfile)

  return model1

def train2(base_loader, val_loader, model1, start_epoch, stop_epoch, params):

  # get optimizer and checkpoint path
  optimizer1 = torch.optim.Adam(model1.gnn1.parameters())


  if not os.path.isdir(params.checkpoint_dir):
    os.makedirs(params.checkpoint_dir)

  # for validation
  global max_acc1
  total_it = 0


  # start
  for epoch in range(start_epoch, stop_epoch):
    model1.train()
    total_it = model1.train_loop_refine1(epoch, base_loader,  optimizer1, total_it) #model are called by reference, no need to return
    model1.eval()

    acc = model1.test_loop_refine1(val_loader)
    if acc > max_acc1:
      print("best model1! save...")
      max_acc1 = acc
      outfile = os.path.join(params.checkpoint_dir, 'best_model_q_s.tar')
      torch.save({'epoch':epoch, 'state':model1.state_dict()}, outfile)
    else:
      print("GG! best accuracy_q_s {:f}".format(max_acc1))

    if ((epoch + 1) % params.save_freq==0) or (epoch==stop_epoch-1):
      outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
      torch.save({'epoch':epoch, 'state':model1.state_dict()}, outfile)

  return model1


# --- main function ---
if __name__=='__main__':

  # set numpy random seed
  np.random.seed(10)
  max_acc1 = 0
  max_acc = 0
  # parser argument
  params = parse_args('train')
  print('--- baseline training: {} ---\n'.format(params.name))
  print(params)

  # output and tensorboard dir
  params.tf_dir = '%s/log/%s'%(params.save_dir, params.name)
  params.checkpoint_dir = '%s/checkpoints/%s'%(params.save_dir, params.name)
  if not os.path.isdir(params.checkpoint_dir):
    os.makedirs(params.checkpoint_dir)

  # dataloader
  print('\n--- prepare dataloader ---')
  if params.dataset == 'multi':
    print('  train with multiple seen domains (unseen domain: {})'.format(params.testset))
    datasets = ['miniImagenet', 'cars', 'places', 'cub', 'plantae']
    datasets.remove(params.testset)
    base_file = [os.path.join(params.data_dir, dataset, 'base.json') for dataset in datasets]
    val_file  = os.path.join(params.data_dir, 'miniImagenet', 'val.json')
  else:
    print('  train with single seen domain {}'.format(params.dataset))
    base_file  = os.path.join(params.data_dir, params.dataset, 'base.json')
    val_file   = os.path.join(params.data_dir, params.dataset, 'val.json')

  # model
  print('\n--- build model ---')
  if 'Conv' in params.model:
    image_size = 84
  else:
    image_size = 224


  print('  baseline training the model {} with feature encoder {}'.format(params.method, params.model))

    #load dataset
  n_query = max(1, int(16* params.test_n_way/params.train_n_way))

  train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot)
  base_datamgr            = SetDataManager(image_size, n_query = n_query,  **train_few_shot_params)
  base_loader             = base_datamgr.get_data_loader( base_file , aug = params.train_aug )

  test_few_shot_params     = dict(n_way = params.test_n_way, n_support = params.n_shot)
  val_datamgr             = SetDataManager(image_size, n_query = n_query, **test_few_shot_params)
  val_loader              = val_datamgr.get_data_loader( val_file, aug = False)

  checkpoint_dir1 = '%s/checkpoints/res18one_TESTSET_ori_METHOD' % (params.save_dir)# change your model name
  modelfile1 = get_best_file(checkpoint_dir1)


  model1           = GnnNet1( model_dict[params.model], tf_path=params.tf_dir, **train_few_shot_params)
  model1.cuda()



  if modelfile1 is not None:
    tmp1 = torch.load(modelfile1)
    try:
      model1.load_state_dict(tmp1['state'])
    except RuntimeError:
      print('warning! RuntimeError when load_state_dict()!')
      model1.load_state_dict(tmp1['state'], strict=False)
    except KeyError:
      for k in tmp1['model_state']:   ##### revise latter
        if 'running' in k:
          tmp1['model_state'][k] = tmp1['model_state'][k].squeeze()
      model1.load_state_dict(tmp1['model_state'], strict=False)
    except:
      raise



  # training
  print('\n--- start the training ---')
  for i in range(10):
    model1 = train1(base_loader, val_loader,  model1, 0, 300, params)
    modelfile_s_q = '%s/checkpoints/%s/best_model_s_q.tar' % (params.save_dir,params.name)

    if modelfile_s_q is not None:
      tmp_s_q = torch.load(modelfile_s_q)
      try:
        model1.load_state_dict(tmp_s_q['state'])
      except RuntimeError:
        print('warning! RuntimeError when load_state_dict()!')
        model1.load_state_dict(tmp_s_q['state'], strict=False)
      except KeyError:
        for k in tmp_s_q['model_state']:  ##### revise latter
          if 'running' in k:
            tmp_s_q['model_state'][k] = tmp_s_q['model_state'][k].squeeze()
        model1.load_state_dict(tmp_s_q['model_state'], strict=False)
      except:
        raise

    model1 = train2(base_loader, val_loader, model1, 0, 300, params)
    modelfile_q_s = '%s/checkpoints/%s/best_model_q_s.tar' % (params.save_dir, params.name)

    if modelfile_q_s is not None:
      tmp_q_s = torch.load(modelfile_q_s)
      try:
        model1.load_state_dict(tmp_q_s['state'])
      except RuntimeError:
        print('warning! RuntimeError when load_state_dict()!')
        model1.load_state_dict(tmp_q_s['state'], strict=False)
      except KeyError:
        for k in tmp_q_s['model_state']:  ##### revise latter
          if 'running' in k:
            tmp_q_s['model_state'][k] = tmp_q_s['model_state'][k].squeeze()
        model1.load_state_dict(tmp_q_s['model_state'], strict=False)
      except:
        raise