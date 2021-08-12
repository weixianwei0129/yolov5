# Loss functions

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyper parameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        """选择符合条件的anchor以及gt box"""
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            # pi: (B, c, h, w)
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj (b, c, h)

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                """计算IOU-loss"""
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]  # wh
                pbox = torch.cat((pxy, pwh), 1)  # predicted box x,y,w,h
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                """给tobj赋值iou的得分, 用于计算目标/背景的分类loss"""
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                # detach = 1.0
                """将预测框和gt-box的iou作为权重 乘到 conf分支, 用于描述预测的质量"""
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # self.gr=1.0 iou ratio

                # Classification
                """多类目标的loss"""
                if self.nc > 1:  # cls loss (only if multiple classes)
                    # self.cp = 1, self.cn = 0
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]

        """给各个loss分配权重"""
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p: list, targets):
        """
        p 是一个列表, 其包含nl个特征层, 例如: 论文用了三层, nl=3
            对于每个层的特征, 其shape是[Batch, C, H, W]
        target shape is (N, 6),
            其中 'N' 表示 一个batch中全部图片的label拼接起来, 也就是一个batch的图片中全部gt-box的数量
            '6' 表示 (img_id: 该gt-box来自哪张图片,其索引值 ,class_id: 这个gt-box的类别, x,y,w,h)
        """
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        """
        =============step1=============
        将targets 重复3遍(3=层anchor数目)，
        也就是将每个gt bbox复制变成独立的3份，
        方便和每个位置的3个anchor单独匹配
        ================================
        """

        """假设na=3, nt=100"""
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to grid space gain

        """ai: (na, nt) 表示 anchor的索引, 用于记录当前gt-box和当前层的哪个anchor匹配"""
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)

        """
        targets: (na, nt, 6), ai: (na, nt, 1) = (na, nt, 7)
        先repeat gt-box 3(na)次, 相当于一个gt-box变成了3(na)个, 然后依次和3(na)个anchor单独匹配.
        """
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        """cell(网格)的中心偏移"""
        g = 0.5  # bias
        """与(0, 0)相邻的四个cell的坐标"""
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m表示的是{j:左边},{k:下},{l:右},{m:上}
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        """对每个特征图进行操作，顺序为降采样8-16-32"""
        for i in range(self.nl):

            anchors = self.anchors[i]

            """p是网络输出的值; 其中p[i].shape: (b, c, h, w); gain 里面装的是 1, 1, w, h, w, h, 1"""
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # gain 的内容是: w h w h

            """targets: na, nt, 7 gain: (1, 1, w, h, w, h, 1) 
            即: (image, class, x, y, w, h, index) * (1, 1, w, h, w, h, 1) 
            获得gt-boxes在当前特征层上的像素坐标. 
            """
            # Match targets to anchors
            t = targets * gain  # (na, nt, 7)(image, class, x, y, w, h, index)

            """
            =============step2=============
            对每个输出层单独匹配。
            target wh和anchor的wh计算比例，
            如果比例过大，则说明匹配度不高，
            将该gt-box过滤， 在当前层认为是bg
            ===============================
            """
            if nt:
                # Matches
                """
                计算长宽比,过滤掉要么太大,要么太小,要么wh差距大的gt-box
                通过过滤后, 会出现某个gt-box仅仅和当前层的某几个anchor匹配
                'j'是要保留的gt-box的索引
                """
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                """
                =============step3=============
                计算最近的2个邻居网格
                ===============================
                """
                gxy = t[:, 2:4]  # grid xy (na, 2) gt-box的中心点坐标 以左上角为原点算坐标
                gxi = gain[[2, 3]] - gxy  # inverse (w, h) - (x, y) 以右下角为原点算坐标
                """选择三个cell, 一个为gt-box中心点落在的当前cell, 
                其次是和当前cell距离最近且相邻的另外两个cell, j k l m必有两个为True
                """
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T  # 左边的cell, 下边的cell
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T  # 右边的cell, 上边的cell
                j = torch.stack((torch.ones_like(j), j, k, l, m))  # (na, 5)
                """上面off预设了5个cell"""
                t = t.repeat((5, 1, 1))[j]  # (5,na,5)[j]
                """选择最近的三个cell"""
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0
            """
            =============step4=============
            对每个gt-box找出对应的正样本anchor.
            其中'b'表示当前gt-box的图片编号，
            'c'表示当前gt-box的类别编号,
            'gxy' 不考虑offset(即yolo-v3中设定的该Bbox的负责预测网格),
            'gwh' 为gt-box的归一化wh,
            'gi,gj'是对应的负责预测该gt-box的网格坐标,
            'a'表示当前gt-box和当前层的第几个anchor匹配上;
            ==============================
            """
            # Define
            b, c = t[:, :2].long().T  # image id, class id
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices
            # Append
            a = t[:, 6].long()  # anchor indices

            """保存: 图像序号, anchor序号, 特征图上的坐标"""
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
