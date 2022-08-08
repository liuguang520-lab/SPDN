import torch
import torch.nn as nn
import numpy as np
from methods.meta_template import MetaTemplate
from methods.gnn import GNN_nl
from methods import backbone
from itertools import combinations_with_replacement

class GnnNet1(MetaTemplate):
  maml=True
  def __init__(self, model_func,  n_way, n_support, tf_path=None):
    super(GnnNet1, self).__init__(model_func, n_way, n_support, tf_path=tf_path)

    # loss function
    self.loss_fn = nn.CrossEntropyLoss()

    # metric function
    self.fc = nn.Sequential(nn.Linear(self.feat_dim, 128), nn.BatchNorm1d(128, track_running_stats=False)) if not self.maml else nn.Sequential(backbone.Linear_fw(self.feat_dim, 128), backbone.BatchNorm1d_fw(128, track_running_stats=False))
    self.gnn = GNN_nl(128 + self.n_way, 96, self.n_way)
    self.gnn1 = GNN_nl(128+self.n_way,96,self.n_way)
    self.method = 'GnnNet'

    # fix label for training the metric function   1*nw(1 + ns)*nw
    support_label = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).unsqueeze(1)
    support_label = torch.zeros(self.n_way*self.n_support, self.n_way).scatter(1, support_label, 1).view(self.n_way, self.n_support, self.n_way)#one-hot
    support_label = torch.cat([support_label, torch.zeros(self.n_way, 1, n_way)], dim=1)
    self.support_label = support_label.view(1, -1, self.n_way)#1*30*5,1*10*5

  def cuda(self):
    self.feature.cuda()
    self.fc.cuda()
    self.gnn.cuda()
    self.gnn1.cuda()
    self.support_label = self.support_label.cuda()
    return self

  def set_forward(self,x,is_feature=False):
    x = x.cuda()

    if is_feature:
      # reshape the feature tensor: n_way * n_s + 15 * f
      assert(x.size(1) == self.n_support + 15)
      z = self.fc(x.view(-1, *x.size()[2:]))
      z = z.view(self.n_way, -1, z.size(1))
    else:
      # get feature using encoder
      x = x.view(-1, *x.size()[2:])
      z = self.fc(self.feature(x))#85*128
      z = z.view(self.n_way, -1, z.size(1))

    # stack the feature for metric function: n_way * n_s + n_q * f -> n_q * [1 * n_way(n_s + 1) * f]
    z_stack = [torch.cat([z[:, :self.n_support], z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(1, -1, z.size(2)) for i in range(self.n_query)]#n_support=shot
    assert(z_stack[0].size(1) == self.n_way*(self.n_support + 1))
    scores,scores1 = self.forward_gnn(z_stack)
    return scores,scores1

  def forward_gnn(self, zs):
    # gnn inp: n_q * n_way(n_s + 1) * f
    nodes = torch.cat([torch.cat([z, self.support_label], dim=2) for z in zs], dim=0)
    scores = self.gnn(nodes)
    medium = torch.zeros_like(nodes)
    for i in range(self.n_way):
      medium[:,2*i:2*i+1,:] = nodes[:,2*i+1:2*i+2,:]
      medium[:,2*i+1:2*i+2,:] = nodes[:,2*i:2*i+1,:]
      medium[:,2*i:2*i+1,128:] = nodes[:,2*i:2*i+1,128:]
      medium[:,2*i+1:2*i+2,128:] = nodes[:,2*i+1:2*i+2,128:]
    scores1 = self.gnn1(medium)

    # n_q * n_way(n_s + 1) * n_way -> (n_way * n_q) * n_way
    scores = scores.view(self.n_query, self.n_way, self.n_support + 1, self.n_way)[:, :, -1].permute(1, 0, 2).contiguous().view(-1, self.n_way)
    scores1 = scores1.view(self.n_query, self.n_way, self.n_support + 1, self.n_way)[:, :, -1].permute(1, 0,2).contiguous().view( -1, self.n_way)

    return scores,scores1

  def set_forward_loss(self, x):
    y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
    y_query = y_query.cuda()



    scores_s_q, scores_q_s = self.set_forward(x)
    loss_s_q = self.loss_fn(scores_s_q, y_query)
    loss_q_s = self.loss_fn(scores_q_s, y_query)

    loss = loss_s_q+loss_q_s
    return scores_s_q, scores_q_s, loss

  def set_forward_refine(self,x,is_feature=False):
    x = x.cuda()

    if is_feature:
      # reshape the feature tensor: n_way * n_s + 15 * f
      assert(x.size(1) == self.n_support + 15)
      z = self.fc(x.view(-1, *x.size()[2:]))
      z = z.view(self.n_way, -1, z.size(1))
    else:
      # get feature using encoder
      x = x.view(-1, *x.size()[2:])
      z = self.fc(self.feature(x))#85*128
      z = z.view(self.n_way, -1, z.size(1))

    # stack the feature for metric function: n_way * n_s + n_q * f -> n_q * [1 * n_way(n_s + 1) * f]
    z_stack = [torch.cat([z[:, :self.n_support], z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(1, -1, z.size(2)) for i in range(self.n_query)]
    assert(z_stack[0].size(1) == self.n_way*(self.n_support + 1))
    scores = self.forward_gnn_refine(z_stack)
    return scores

  def forward_gnn_refine(self, zs):
    # gnn inp: n_q * n_way(n_s + 1) * f
    nodes = torch.cat([torch.cat([z, self.support_label], dim=2) for z in zs], dim=0)
    scores = self.gnn(nodes)

    scores_q = torch.zeros(self.n_query,self.n_way,self.n_way)#choose query
    for s in range(self.n_way):
      scores_q[:,s:s+1,:] = scores[:,self.n_support+(self.n_support+1)*s:self.n_support+1+(self.n_support+1)*s,:]



    label = torch.Tensor([0,1,2,3,4])
    z=[]

    for p in combinations_with_replacement(label, self.n_way):
      z.append(p)
    z = torch.Tensor(z)

    z=np.array(z)
    z=torch.from_numpy(z)
    z=torch.nn.functional.one_hot(z.long(),self.n_way)

    q_label_all =torch.zeros(self.n_query,self.n_way,self.n_way)
    for q in range(self.n_query):
      distance = []
      for e in range(z.size(0)):
        dis = ((scores_q[q]-z[e])**2).sum()
        distance.append(dis)
      distance = torch.Tensor(distance).cuda()

      distance_sort = distance.sort()[1][:5]

      chose_label = torch.zeros(5,self.n_way,self.n_way)
      for l in range(5):
        chose_label[l] = z[distance_sort[l]]
      chose_label=torch.Tensor(chose_label).cuda()

      zero = torch.zeros_like(scores_q)

      loss_q_s=[]

      # value choosed label
      for c in range(5):
        medium = torch.clone(nodes)
        for i in range(self.n_way):
          medium[:, self.n_support - 1 + (self.n_support + 1) * i:self.n_support + (self.n_support + 1) * i, :] = nodes[:,self.n_support + (self.n_support + 1) * i:self.n_support+1+(self.n_support + 1) * i, :]
          medium[:, self.n_support + (self.n_support + 1) * i:self.n_support +1 + (self.n_support + 1) * i, :] = nodes[:,self.n_support - 1 + (self.n_support + 1) * i:self.n_support + (self.n_support + 1) * i, :]
          medium[:, self.n_support + (self.n_support + 1) * i:self.n_support+1 + (self.n_support + 1) * i, 128:] = zero[:, i : i+1, :]
          medium[:, self.n_support - 1 + (self.n_support + 1) * i:self.n_support + (self.n_support + 1) * i, 128:] = scores_q[:, i:i + 1, :]
          medium[q:q+1,self.n_support - 1 + (self.n_support + 1) * i:self.n_support + (self.n_support + 1) * i,128:]=chose_label[c:c+1,i:i+1,:]
        scores1 = self.gnn1(medium)
        scores1 = scores1.view(self.n_query, self.n_way, self.n_support + 1, self.n_way)[:, :, -1].permute(1, 0,2).contiguous().view(-1, self.n_way)
        scores_zero = torch.zeros(self.n_way,self.n_way)
        scores_zero = scores_zero.cuda()

        for s in range(self.n_way):
          scores_zero[s:s+1,:]=scores1[s*self.n_query+q:self.n_query*s+1+q,:]

        scores_zero = scores_zero.argmax(dim=1).cpu()
        loss = ((scores_zero-label)**2).sum()
        loss_q_s.append(loss)
      loss_q_s = torch.tensor(loss_q_s).cuda()
      loss_index = loss_q_s.argmin()
      q_label = z[distance_sort[loss_index]]
      q_label_all[q]=q_label
    scores = q_label_all.permute(1, 0, 2).contiguous().view(-1, self.n_way)
    return scores

  def set_forward_loss_refine(self, x): #x:5*21*3*224*224,5*17*3*224*224
    y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
    y_query = y_query.cuda()

    scores_s_q, scores_q_s = self.set_forward_refine(x)
    loss_s_q = self.loss_fn(scores_s_q, y_query)
    loss_q_s = self.loss_fn(scores_q_s, y_query)

    loss = loss_s_q+loss_q_s
    return scores_s_q, scores_q_s, loss,loss_s_q,loss_q_s
