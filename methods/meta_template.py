import torch.nn as nn
import numpy as np
from abc import abstractmethod
from tensorboardX import SummaryWriter
import torch

class MetaTemplate(nn.Module):
  def __init__(self, model_func, n_way, n_support, flatten=True, leakyrelu=False, tf_path=None, change_way=True):
    super(MetaTemplate, self).__init__()
    self.n_way      = n_way
    self.n_support  = n_support
    self.n_query    = -1 #(change depends on input)
    self.feature    = model_func(flatten=flatten, leakyrelu=leakyrelu)
    self.feat_dim   = self.feature.final_feat_dim
    self.change_way = change_way  #some methods allow different_way classification during training and test
    self.tf_writer = SummaryWriter(log_dir=tf_path) if tf_path is not None else None

  @abstractmethod
  def set_forward(self,x,is_feature):
    pass

  @abstractmethod
  def set_forward_loss(self, x):
    pass

  @abstractmethod
  def set_forward_loss_refine(self, x):
    pass

  @abstractmethod
  def set_forward_refine(self, x, is_feature=False):
    pass

  @abstractmethod
  def set_forward_loss_refine1(self, x):
    pass

  @abstractmethod
  def set_forward_refine1(self, x, is_feature=False):
    pass

  def forward(self,x):
    out  = self.feature.forward(x)
    return out

  def parse_feature(self,x,is_feature):
    x = x.cuda()
    if is_feature:
      z_all = x
    else:
      x           = x.contiguous().view( self.n_way * (self.n_support + self.n_query), *x.size()[2:])
      z_all       = self.feature.forward(x)
      z_all       = z_all.view( self.n_way, self.n_support + self.n_query, -1)
    z_support   = z_all[:, :self.n_support]
    z_query     = z_all[:, self.n_support:]

    return z_support, z_query

  def correct(self, x):
    # scores, loss ,_ ,_= self.set_forward_loss(x)
    scores_s_q,scores_q_s, loss= self.set_forward_loss(x)
    y_query = np.repeat(range( self.n_way ), self.n_query )

    topk_scores, topk_labels = scores_s_q.data.topk(1, 1, True, True)
    topk_ind = topk_labels.cpu().numpy()
    top1_correct = np.sum(topk_ind[:,0] == y_query)

    topk_scores, topk_labels1 = scores_q_s.data.topk(1, 1, True, True)
    topk_ind1 = topk_labels1.cpu().numpy()
    top1_correct1 = np.sum(topk_ind1[:,0] == y_query)
    return float(top1_correct),float(top1_correct1), len(y_query), loss.item()*len(y_query)


  def train_loop(self, epoch, train_loader, optimizer,total_it):
    print_freq = len(train_loader) // 10

    avg_loss=0
    for i, (x,_ ) in enumerate(train_loader):#_:n_way,i:100
      self.n_query = x.size(1) - self.n_support
      if self.change_way:
        self.n_way  = x.size(0)
      optimizer.zero_grad()
      scores_s_q, scores_q_s, loss_s_q = self.set_forward_loss(x)
      loss_s_q.backward()
      optimizer.step()
      avg_loss = avg_loss+loss_s_q.item() #tensor value


      if (i + 1) % print_freq == 0:
        print('Epoch {:d} | Batch {:d}/{:d} | avg_Loss {:f}'.format(epoch, i + 1,len(train_loader),avg_loss / float(i + 1)))
      if (total_it + 1) % 10 == 0 and self.tf_writer is not None:
        self.tf_writer.add_scalar(self.method + '/query_loss', loss_s_q.item(), total_it + 1)
      total_it += 1
    return total_it

  def train_loop_refine(self, epoch, train_loader, optimizer,total_it):
    print_freq = len(train_loader) // 10

    avg_loss=0
    avg_loss_s_q=0
    avg_loss_q_s = 0
    for i, (x,_ ) in enumerate(train_loader):#_:n_way,i:100
      self.n_query = x.size(1) - self.n_support
      if self.change_way:
        self.n_way  = x.size(0)
      optimizer.zero_grad()
      scores_s_q, scores_q_s, loss,loss_s_q,loss_q_s = self.set_forward_loss_refine(x)
      loss.backward()
      optimizer.step()
      avg_loss = avg_loss+loss.item() #tensor value
      avg_loss_s_q = avg_loss_s_q+loss_s_q.item()
      avg_loss_q_s = avg_loss_q_s+loss_q_s.item()

      if (i + 1) % print_freq == 0:
        print('Epoch {:d} | Batch {:d}/{:d} | avg_Loss {:f}, avg_loss_s_q {:f}, avg_loss_q_s {:f}'.format(epoch, i + 1,len(train_loader),avg_loss / float(i + 1),avg_loss_s_q / float(i+1),avg_loss_q_s/float(i+1)))
      if (total_it + 1) % 10 == 0 and self.tf_writer is not None:
        self.tf_writer.add_scalar(self.method + '/query_loss', loss.item(), total_it + 1)
      total_it += 1
    return total_it

  def train_loop_refine1(self, epoch, train_loader, optimizer,total_it):
    print_freq = len(train_loader) // 10

    avg_loss=0
    avg_loss_s_q=0
    avg_loss_q_s = 0
    for i, (x,_ ) in enumerate(train_loader):#_:n_way,i:100
      self.n_query = x.size(1) - self.n_support
      if self.change_way:
        self.n_way  = x.size(0)
      optimizer.zero_grad()
      scores_s_q, scores_q_s, loss,loss_s_q,loss_q_s = self.set_forward_loss_refine1(x)
      loss.backward()
      optimizer.step()
      avg_loss = avg_loss+loss.item() #tensor value
      avg_loss_s_q = avg_loss_s_q+loss_s_q.item()
      avg_loss_q_s = avg_loss_q_s+loss_q_s.item()

      if (i + 1) % print_freq == 0:
        print('Epoch {:d} | Batch {:d}/{:d} | avg_Loss {:f}, avg_loss_s_q {:f}, avg_loss_q_s {:f}'.format(epoch, i + 1,len(train_loader),avg_loss / float(i + 1),avg_loss_s_q / float(i+1),avg_loss_q_s/float(i+1)))
      if (total_it + 1) % 10 == 0 and self.tf_writer is not None:
        self.tf_writer.add_scalar(self.method + '/query_loss', loss.item(), total_it + 1)
      total_it += 1
    return total_it

  def test_loop(self, test_loader, record = None):
    loss = 0.
    count = 0
    acc_all = []
    acc_all1 = []
    iter_num = len(test_loader)
    for i, (x,_) in enumerate(test_loader):

      if self.change_way:
        self.n_way  = x.size(0)
      correct_this,correct_this1, count_this, loss_this = self.correct(x)
      acc_all.append(correct_this/ count_this*100  )
      acc_all1.append(correct_this1 / count_this * 100)
      loss += loss_this
      count += count_this

    acc_all  = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std  = np.std(acc_all)
    print('--- %d Loss = %.6f ---' %(iter_num,  loss/count))
    print('--- %d Test Acc = %4.2f%% +- %4.2f%% ---' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))

    acc_all1  = np.asarray(acc_all1)
    acc_mean1 = np.mean(acc_all1)
    acc_std1  = np.std(acc_all1)
    print('--- %d Loss = %.6f ---' %(iter_num,  loss/count))
    print('--- %d Test Acc = %4.2f%% +- %4.2f%% ---' %(iter_num,  acc_mean1, 1.96* acc_std1/np.sqrt(iter_num)))

    return acc_mean


  def correct_refine(self, x):
    # scores, loss ,_ ,_= self.set_forward_loss(x)
    scores_s_q,scores_q_s, loss,loss_s_q,loss_q_s= self.set_forward_loss_refine(x)
    y_query = np.repeat(range( self.n_way ), self.n_query )

    topk_scores, topk_labels = scores_s_q.data.topk(1, 1, True, True)
    topk_ind = topk_labels.cpu().numpy()
    top1_correct = np.sum(topk_ind[:,0] == y_query)

    return float(top1_correct), len(y_query), loss.item()*len(y_query)

  def test_loop_refine(self, test_loader, record = None):
    loss = 0.
    count = 0
    acc_all = []

    iter_num = len(test_loader)
    for i, (x,_) in enumerate(test_loader):
      # self.n_query = 16
      if self.change_way:
        self.n_way  = x.size(0)
      correct_this, count_this, loss_this = self.correct_refine(x)
      acc_all.append(correct_this/ count_this*100  )
      loss += loss_this
      count += count_this

    acc_all  = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std  = np.std(acc_all)
    print('--- %d Loss = %.6f ---' %(iter_num,  loss/count))
    print('--- %d Test Acc = %4.2f%% +- %4.2f%% ---' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))

    return acc_mean

  def correct_refine1(self, x):
    # scores, loss ,_ ,_= self.set_forward_loss(x)
    scores_s_q, scores_q_s, loss, loss_s_q, loss_q_s = self.set_forward_loss_refine1(x)
    y_query = np.repeat(range(self.n_way), self.n_query)

    topk_scores, topk_labels = scores_q_s.data.topk(1, 1, True, True)
    topk_ind = topk_labels.cpu().numpy()
    top1_correct = np.sum(topk_ind[:, 0] == y_query)

    return float(top1_correct), len(y_query), loss.item() * len(y_query)

  def test_loop_refine1(self, test_loader, record=None):
    loss = 0.
    count = 0
    acc_all = []

    iter_num = len(test_loader)
    for i, (x, _) in enumerate(test_loader):
      # self.n_query = 16
      if self.change_way:
        self.n_way = x.size(0)
      correct_this, count_this, loss_this = self.correct_refine1(x)
      acc_all.append(correct_this / count_this * 100)
      loss += loss_this
      count += count_this

    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    print('--- %d Loss = %.6f ---' % (iter_num, loss / count))
    print('--- %d Test Acc = %4.2f%% +- %4.2f%% ---' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

    return acc_mean