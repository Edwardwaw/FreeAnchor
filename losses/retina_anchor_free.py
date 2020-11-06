import torch
from utils.retinanet import BoxCoder
from commons.boxs_utils import box_iou
import torch.nn.functional as F
# from losses.commons import smooth_l1_loss,giou


def get_world_size():
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()


# def reduce_sum(loss):
#     import torch.distributed as dist
#     world_size = get_world_size()
#     if world_size <= 1:
#         return loss
#     loss_op=loss
#     with torch.no_grad():
#         dist.reduce(loss_op, dst=0)
#         if dist.get_rank() == 0:
#             # only main process gets accumulated, so only divide by
#             # world_size in this case
#             loss_op /= world_size
#     return loss_op

def reduce_sum(tensor):
    import torch.distributed as dist
    g_num = get_world_size()
    if g_num <= 1:
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor



# def mean_max(x):
#     '''
#
#     :param x:  [gt, tok_anchor]
#     :return:
#     '''
#     weights=1/(1.-x).clamp(min=1e-10)
#     weights /= weights.sum(-1)[:,None]
#     bag_prob = (weights*x).sum(-1)
#     return -bag_prob.clamp(min=1e-10, max=1 - 1e-10).log()
#
#
#
#
#
# class RetinaAnchorFreeLoss(object):
#     def __init__(self,gamma=2.0,alpha=0.25,top_k=64,box_iou_thresh=0.6,box_reg_weight=0.75,beta=1./9):
#         super(RetinaAnchorFreeLoss, self).__init__()
#         self.gamma=gamma
#         self.alpha=alpha
#         self.top_k=top_k
#         self.box_iou_thresh=box_iou_thresh
#         self.box_reg_weight=box_reg_weight
#         self.beta=beta
#         self.box_coder=BoxCoder()
#
#     def __call__(self, cls_predicts, box_predicts, anchors, targets):
#         '''
#
#         :param cls_predicts:
#         :param box_predicts:
#         :param anchors:
#         :param targets:
#         :return:
#         '''
#
#         device=cls_predicts[0].device
#         bs=cls_predicts[0].shape[0]
#         cls_num=cls_predicts[0].shape[-1]
#         expand_anchor=torch.cat(anchors,dim=0)
#
#         positive_loss_list=list()
#         negative_loss_list = list()
#
#
#         for bi in range(bs):
#             batch_cls_predicts = torch.cat([cls_item[bi] for cls_item in cls_predicts], dim=0) \
#                 .sigmoid() \
#                 .clamp(min=1e-6, max=1 - 1e-6)   # cls_predict, shape=[num_anchors,80]
#             batch_targets = targets[targets[:, 0] == bi, 1:]  # gt_box, shape=[num_gts,6]  6==>conf_score,label_id,x1,y1,x2,y2
#
#             # if no gt_box exist, just calc focal loss in negative condition
#             if len(batch_targets) == 0:
#                 negative_loss = -(1 - self.alpha) * (batch_cls_predicts ** self.gamma) * (1 - batch_cls_predicts).log()
#                 negative_loss_list.append(negative_loss.sum())
#                 continue
#
#             batch_box_predicts = torch.cat([box_item[bi] for box_item in box_predicts], dim=0)   # box_predict , shape=[num_anchors,4]
#
#
#             #calc positive loss
#             targets_anchor_iou=box_iou(batch_targets[:,2:],expand_anchor)   # shape=(num_gts,num_anchors)
#             _,top_k_anchor_idx=targets_anchor_iou.topk(k=self.top_k,dim=1,sorted=False)  # shape=(num_gts,top_k)   元素的取值范围[0,num_gts) 表示匹配到某个gt的anchor集合的索引
#
#             # one question here: match_cls_prob = P cls (right?)
#             matched_cls_prob = batch_cls_predicts[top_k_anchor_idx].gather(dim=-1, index=(
#                 batch_targets[:, [1]][:, None, :]).long().repeat(1, self.top_k, 1)).squeeze(-1)    # shape=(num_gts,top_k) 元素的取值范围[0,num_cls) 表示匹配到某个gt的anchor所属的类别
#             match_box_target = self.box_coder.encoder(expand_anchor[top_k_anchor_idx], batch_targets[:, None, 2:])   # shape=(num_gts,top_k,4) 元素表示匹配到某个gt的anchor的box回归目标编码
#             # P loc
#             matched_box_prob=(-self.box_reg_weight*smooth_l1_loss(batch_box_predicts[top_k_anchor_idx],match_box_target,self.beta)).sum(-1).exp()   # Ploc shape=[num_gts,top_k]
#
#             positive_loss = self.alpha * mean_max(matched_cls_prob * matched_box_prob).sum()
#             positive_loss_list.append(positive_loss)
#
#
#             #calc negative loss
#             with torch.no_grad():
#                 box_localization=self.box_coder.decoder(batch_box_predicts,expand_anchor)  # shape=[num_anchors,4]  decode predicted box into original input size
#
#             target_box_iou=box_iou(batch_targets[:, 2:], box_localization)  # shape=[num_gts,num_anchors]
#             t1=self.box_iou_thresh
#             t2=target_box_iou.max(dim=1,keepdim=True)[0].clamp(min=t1 + 1e-6)  # shape=[num_gts,1]
#             # 反映anchor是关于某个gt_box的正样本的矩阵,取值范围为[0,1],且每一行有且仅有1个元素的取值为1, shape=[num_gts,num_anchors]
#             target_box_prob=((target_box_iou-t1)/(t2-t1)).clamp(min=0.,max=1.)
#
#             # shape=[2,num_gts]  第0行元素代表所对应的gt_box的索引, 第1行元素代表所对应的gt_box所属的类别
#             indices=torch.stack([torch.arange(len(batch_targets), device=device),batch_targets[:,1]],dim=0).long()
#             '''
#             shape=[num_gts, max_cls_id+1, num_anchors] 按照类别的取值填充
#             note： 如果索引为gt_id的gt_box所属的类别为label_id, 则object_cls_box_prob[gt_id,label_id]=target_box_prob[gt_id], 其他位置均为0
#             '''
#             object_cls_box_prob=torch.sparse_coo_tensor(indices,target_box_prob,device=device)
#
#             # cls_idx, anchor_idx.shape=[n]
#             cls_idx, anchor_idx = torch.sparse.sum(object_cls_box_prob, dim=0).to_dense().nonzero(as_tuple=False).t()
#             if len(cls_idx) == 0:
#                 negative_loss = -(1 - self.alpha) * (batch_cls_predicts ** self.gamma) * (1 - batch_cls_predicts).log()
#                 negative_loss_list.append(negative_loss.sum())
#                 continue
#             # P{aj<A-}
#             anchor_positive_max_prob = torch.where(
#                 batch_targets[:, [1]].long() == cls_idx,  # shape=[num_gts,n]
#                 target_box_prob[:, anchor_idx],
#                 torch.tensor(data=0., device=device)
#             ).max(dim=0)[0]
#
#             anchor_cls_assign_prob = torch.zeros(size=(len(expand_anchor), cls_num), device=device)  # shape=[num_anchors,num_cls]
#             anchor_cls_assign_prob[anchor_idx, cls_idx] = anchor_positive_max_prob
#             # negative_prob = []
#             negative_prob = batch_cls_predicts * (1 - anchor_cls_assign_prob)
#             negative_loss = -(1 - self.alpha) * (negative_prob ** self.gamma) * (1 - negative_prob).log()
#             negative_loss_list.append(negative_loss.sum())
#
#         # negative_losses = torch.stack(negative_loss_list).sum() / max(1, len(targets)) / self.top_k
#         negative_losses = torch.stack(negative_loss_list).sum() / max(1, len(targets))
#         if len(positive_loss_list) == 0:
#             total_loss = negative_losses
#             return total_loss, torch.stack([negative_losses, torch.tensor(data=0., device=device)]), len(targets)
#
#         positive_losses = torch.stack(positive_loss_list).sum() / max(1, len(targets))
#         total_loss = negative_losses + positive_losses
#         return total_loss, torch.stack([negative_losses, positive_losses]), len(targets)








def smooth_l1_loss(pred, target, weight, beta):
    '''
    calc loc_loss=weight*L_ij(loc)
    e{-loc_loss}=P(loc)

    params
    :param pred: predicted box, shape=[num_gts,topk,4]
    :param target:  target box, shape=[num_gts,topk,4]
    :param weight:  hyper param for loc loss, P_ij(loc)=e_{-beta*L_ij(loc)},  L_ij(loc)=smooth_l1_loss, beta=weight
    :param beta:  hyper param for smooth l1 loss, default=1.0/9
    :return:
    loss: shape=[num_gts,topk]
    '''
    abs_val=torch.abs(target-pred)
    cond=abs_val<beta
    loss=torch.where(cond,0.5*abs_val**2/beta,abs_val-0.5*beta).sum(dim=-1)
    loss=weight*loss
    return loss



def bce_loss(y_pred,y_true,reduction=None):
    my_loss = - y_true * torch.log(y_pred) - (1 - y_true) * torch.log(1 - y_pred)
    return my_loss


##  -log(mean_max(logits))
def positive_bag_loss(logits,*args,**kwargs):
    '''

    :param logits: P_cls*P_loc, shape=[gt, topk]
    :param args:
    :param kwargs:
    :return:
    '''
    # bag_prob = Mean-max(logits)
    weight = 1./(1.-logits).clamp(min=1e-12)
    weight /= weight.sum(*args, **kwargs).unsqueeze(dim=-1)
    bag_prob=(weight*logits).sum(*args, **kwargs)
#     return bce_loss(bag_prob.clamp(min=1e-12,max=1.0 - 1e-12),torch.ones_like(bag_prob))
    return -(bag_prob.clamp(min=1e-10,max=1.0 - 1e-10).log())




def focal_loss(logits,gamma):
    '''

    :param logits: P{A_}*(1-P_bg), P{A_}=loc conf, (1-P_bg)=cls conf=P_cls,  shape=[num_anchors,num_cls]
    :param gamma:
    :return:
    '''
#     return torch.sum((logits ** gamma) * bce_loss(logits.clamp(min=1e-12,max=1.0 - 1e-12), torch.zeros_like(logits)))
    bce_loss = - (1. - logits.clamp(min=1e-10, max=1.0 - 1e-10)).log()
    bce_loss = -1*((1. - logits).clamp(min=1e-10, max=1.0 - 1e-10).log().clamp(min=-1000., max=1000.))
    return torch.sum((logits ** gamma) * bce_loss)








class RetinaAnchorFreeLoss(object):
    def __init__(self,gamma=2.0,alpha=0.25,top_k=50,box_iou_thresh=0.6,box_reg_weight=0.75,beta=1./9):
        super(RetinaAnchorFreeLoss, self).__init__()
        self.gamma=gamma
        self.alpha=alpha
        self.top_k=top_k
        self.box_iou_thresh=box_iou_thresh
        self.box_reg_weight=box_reg_weight
        self.beta=beta

        self.positive_bag_loss_func=positive_bag_loss
        self.negative_bag_loss_func=focal_loss
        self.box_coder=BoxCoder()


    def __call__(self, cls_predicts, box_predicts, anchors, targets):
        '''

        :param cls_predicts:
        :param box_predicts:
        :param anchors:
        :param targets:
        :return:
        '''

        device=cls_predicts[0].device
        bs=cls_predicts[0].shape[0]
        cls_num=cls_predicts[0].shape[-1]
        expand_anchor=torch.cat(anchors,dim=0)  #shape=[num_anchors,4]

        positive_numels = 0   # gt_box的数量
        box_prob = list()     # store P_A+,  P_A-=1-P_A+
        positive_loss_list=list()
        negative_loss_list=list()
        cls_probs=list()



        for bi in range(bs):
            cls_prob = torch.cat([cls_item[bi] for cls_item in cls_predicts], dim=0).sigmoid().clamp(min=1e-6, max=1 - 1e-6)   # cls_predict, shape=[num_anchors,80]
            target = targets[targets[:, 0] == bi, 1:]  # gt_box, shape=[num_gts,6]  6==>conf_score,label_id,x1,y1,x2,y2

            # if no gt_box exist, just calc focal loss in negative condition
            if len(target) == 0:
#                 negative_loss = -(cls_prob ** self.gamma) * (1 - cls_prob).log()
                negative_loss = -(cls_prob ** self.gamma) * ((1 - cls_prob).clamp(min=1e-10, max=1.0 - 1e-10).log().clamp(min=-1000., max=1000.))
                negative_loss_list.append(negative_loss.sum())
                continue

            cls_probs.append(cls_prob)
            box_regression = torch.cat([box_item[bi] for box_item in box_predicts],dim=0)  # box_predict , shape=[num_anchors,4]




            with torch.set_grad_enabled(False):

                # box_localization: a_{j}^{loc}, shape: [j, 4]
                box_localization = self.box_coder.decoder(box_regression, expand_anchor)  # shape=[num_anchors,4]  4==>x1,y1,x2,y2

                # object_box_iou: IoU_{ij}^{loc}, shape: [i, j]
                object_box_iou = box_iou(target[:,2:],box_localization)   # shape=(num_gts,num_anchors)

                t1=self.box_iou_thresh
                t2=object_box_iou.max(dim=1,keepdim=True)[0].clamp(min=t1 + 1e-12)  # shape=[num_gts,1]

                # object_box_prob: P{a_{j} -> b_{i}}, shape: [i, j]
                object_box_prob = ((object_box_iou - t1) / (t2 - t1)).clamp(min=0, max=1.)


                '''
                indices.shape=[2,num_gts]
                第0行元素代表所对应的gt_box的索引, 第1行元素代表所对应的gt_box所属的类别
                '''
                indices = torch.stack([torch.arange(len(target),device=device), target[:,1]], dim=0).long()

                # object_cls_box_prob: P{a_{j} -> b_{i}}, shape: [i, c, j]
                '''
                object_cls_box_prob.shape=[num_gts, max_cls_id+1, num_anchors] 按照类别的取值填充
                note： 如果索引为gt_id的gt_box所属的类别为label_id, 则object_cls_box_prob[gt_id,label_id]=target_box_prob[gt_id], 其他位置均为0
                '''
                object_cls_box_prob = torch.sparse_coo_tensor(indices, object_box_prob, device=device)


                """
                image_box_prob: P{a_{j} \in A_{+}}, shape: [j, c] or [num_anchors,num_cls]
                image_box_prob是用来判断一个anchor是否可以匹配到某个目标(无论类别和匹配到gt box是什么)的置信度

                from "start" to "end" implement:
                image_box_prob = torch.sparse.max(object_cls_box_prob, dim=0).t()
                """
                # start

                # indices = torch.nonzero(torch.sparse.sum(object_cls_box_prob, dim=0).to_dense()).t_()  # shape=[2,N]
                indices = torch.sparse.sum(object_cls_box_prob, dim=0).to_dense().nonzero(as_tuple=False).t() # shape=[2,N]


                if indices.numel()==0:
                    image_box_prob = torch.zeros(expand_anchor.shape[0], cls_num).type_as(object_box_prob)
                else:
                    nonzero_box_prob = torch.where(
                        target[:,1].unsqueeze(dim=-1) == indices[0],  # （num_gts,1）== (N) ===>(num_gts,N)
                        object_box_prob[:, indices[1]],
                        torch.tensor([0]).type_as(object_box_prob)
                    ).max(dim=0)[0]  # ===> (N)


                    image_box_prob = torch.sparse_coo_tensor(
                        indices.flip([0]), nonzero_box_prob,
                        size=(expand_anchor.shape[0], cls_num),  # shape=[num_anchors,num_cls]
                        device=device
                    ).to_dense()
                # end
                box_prob.append(image_box_prob)

            # construct bags for objects
            match_quality_matrix = box_iou(target[:,2:], expand_anchor)
            _, matched = torch.topk(match_quality_matrix, self.top_k, dim=1, sorted=False)  # shape=(num_gts,top_k)   元素的取值范围[0,num_gts) 表示匹配到某个gt的anchor集合的索引
            del match_quality_matrix

            # matched_cls_prob: P_{ij}^{cls}
            # shape=(num_gts,top_k) 元素的取值范围[0,num_cls) 表示匹配到某个gt的anchor所属的类别
            matched_cls_prob = cls_prob[matched].gather(dim=-1, index=(target[:, [1]][:, None, :]).long().repeat(1, self.top_k, 1)).squeeze(-1)

            # matched_box_prob: P_{ij}^{loc}
            matched_object_targets = self.box_coder.encoder(expand_anchor[matched], target[:,2:].unsqueeze(dim=1))  # shape=[num_gts,topk,4]
            # P_loc
            retinanet_regression_loss = smooth_l1_loss(box_regression[matched], matched_object_targets, self.box_reg_weight, self.beta)
            matched_box_prob = torch.exp(-retinanet_regression_loss)

            # positive_losses: { -log( Mean-max(P_{ij}^{cls} * P_{ij}^{loc}) ) }
            positive_numels += len(target)
            positive_loss_list.append(self.positive_bag_loss_func(matched_cls_prob * matched_box_prob, dim=1))

        # positive_loss: \sum_{i}{ -log( Mean-max(P_{ij}^{cls} * P_{ij}^{loc}) ) } / ||B||
        # positive_loss = torch.cat(positive_loss_list).sum() / max(1, positive_numels)
        item1=torch.cat(positive_loss_list).sum()
        item2=max(1, positive_numels)
        positive_loss = reduce_sum(item1)/reduce_sum(torch.tensor(data=item2,device=device).float()).item()

        # box_prob: P{a_{j} \in A_{+}}
        box_prob = torch.stack(box_prob, dim=0)
        cls_probs = torch.stack(cls_probs,dim=0)

        # negative_loss: \sum_{j}{ FL( (1 - P{a_{j} \in A_{+}}) * (1 - P_{j}^{bg}) ) } / n||B||
        '''
        (1-P_bg)<==>P_cls   shape=[num_anchors,num_cls]
        P{A-}<==>(1-P{box_cls})
        '''
        if len(negative_loss_list)!=0:
            neg_loss_empty=torch.stack(negative_loss_list,dim=0).sum()
        else:
            neg_loss_empty=0

        # negative_loss = (neg_loss_empty + self.negative_bag_loss_func(cls_probs * (1 - box_prob), self.gamma)) / max(1, positive_numels * self.top_k)
        item3 = neg_loss_empty + self.negative_bag_loss_func(cls_probs * (1 - box_prob), self.gamma)
        item4 = max(1, positive_numels * self.top_k)
        negative_loss = reduce_sum(item3)/reduce_sum(torch.tensor(data=item4,device=device).float()).item()


        total_loss=positive_loss * self.alpha + negative_loss * (1 - self.alpha)
        # total_loss=reduce_sum(total_loss)/get_world_size()
        return total_loss, torch.stack([negative_loss, positive_loss]), positive_numels

