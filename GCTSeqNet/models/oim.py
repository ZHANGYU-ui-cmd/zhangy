import torch
import torch.nn.functional as F
from torch import autograd, nn

# from utils.distributed import tensor_gather

def cos_similarity(inputs, lut):
    inputs_norm = torch.norm(inputs, dim=1, keepdim=True)
    # 计算lut的范数
    lut_norm = torch.norm(lut, dim=1)
    # 计算inputs和lut的点积
    dot_product = torch.mm(inputs, lut.t())
    # 计算余弦相似度
    cosine_sim = dot_product / (inputs_norm * lut_norm + 1e-8)
    return cosine_sim

class OIM(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, targets, lut, cq, header, momentum):
        ctx.save_for_backward(inputs, targets, lut, cq, header, momentum)
        outputs_labeled = inputs.mm(lut.t())
        outputs_unlabeled = inputs.mm(cq.t())
        # outputs_labeled = cos_similarity(inputs, lut)
        # outputs_unlabeled = cos_similarity(inputs, cq)
        return torch.cat([outputs_labeled, outputs_unlabeled], dim=1)

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets, lut, cq, header, momentum = ctx.saved_tensors

        # inputs, targets = tensor_gather((inputs, targets))

        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(torch.cat([lut, cq], dim=0))
            if grad_inputs.dtype == torch.float16:
                grad_inputs = grad_inputs.to(torch.float32)

        for x, y in zip(inputs, targets):
            if y < len(lut):
                lut[y] = momentum * lut[y] + (1.0 - momentum) * x
                lut[y] /= lut[y].norm()
            else:
                cq[header] = x
                header = (header + 1) % cq.size(0)
        return grad_inputs, None, None, None, None, None


def oim(inputs, targets, lut, cq, header, momentum=0.5):
    return OIM.apply(inputs, targets, lut, cq, torch.tensor(header), torch.tensor(momentum))


class OIMLoss(nn.Module):
    def __init__(self, num_features, num_pids, num_cq_size, oim_momentum, oim_scalar):
        super(OIMLoss, self).__init__()
        self.num_features = num_features
        self.num_pids = num_pids
        self.num_unlabeled = num_cq_size
        self.momentum = oim_momentum
        self.oim_scalar = oim_scalar

        self.register_buffer("lut", torch.zeros(self.num_pids, self.num_features))
        self.register_buffer("cq", torch.zeros(self.num_unlabeled, self.num_features))

        self.header_cq = 0

    def forward(self, inputs, roi_label):
        # merge into one batch, background label = 0
        targets = torch.cat(roi_label)
        label = targets - 1  # background label = -1

        inds = label >= 0
        label = label[inds]
        inputs = inputs[inds.unsqueeze(1).expand_as(inputs)].view(-1, self.num_features)

        projected = oim(inputs, label, self.lut, self.cq, self.header_cq, momentum=self.momentum)
        projected *= self.oim_scalar

        self.header_cq = (
            self.header_cq + (label >= self.num_pids).long().sum().item()
        ) % self.num_unlabeled
        loss_oim = F.cross_entropy(projected, label, ignore_index=5554)
        loss_oim = torch.nan_to_num(loss_oim)
        return loss_oim


class LOIM(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, targets, lut, lut_tmp, lut_ious, cq, header, momentum, ious, eps):
        ctx.save_for_backward(inputs, targets, lut, lut_tmp, lut_ious, cq, header, momentum, ious, eps)
        outputs_labeled = inputs.mm(lut.t())
        outputs_unlabeled = inputs.mm(cq.t())
        return torch.cat([outputs_labeled, outputs_unlabeled], dim=1)

    @staticmethod
    def backward(ctx,grad_outputs):
        inputs, targets, lut, lut_tmp, lut_ious, cq, header, momentum, ious, eps = ctx.saved_tensors
        eps = eps.to(ious.device)
        ious = torch.clamp(ious, max=1-eps)

        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(torch.cat([lut, cq], dim=0))
            if grad_inputs.dtype == torch.float16:
                grad_inputs = grad_inputs.to(torch.float32)

        for x, y, s in zip(inputs, targets, ious.view(-1)):
            if y < len(lut):
                indexes = torch.nonzero(lut_ious[y] == -1.0).reshape(-1)
                if len(indexes) == 0:
                    # 进行组间动量计算（参数暂时取0.5）
                    lut_alpha = 0.5
                    if torch.all(lut[y].eq(0)):
                        lut[y] = lut_tmp[y]
                    else:
                        lut[y] = (1.0 - lut_alpha) * lut[y] + lut_alpha * lut_tmp[y]
                        lut[y] /= lut[y].norm()

                    lut_ious[y] = -1
                    lut_ious[y][0] = s
                    lut_tmp[y] = x

                else:
                    # 进行组内平均计算（基于iou）
                    index = indexes[0]
                    lut_ious[y][index] = s
                    if index == 0:
                        lut_tmp[y] = x
                    else:
                        # lut[y] = (1.0 - s) * lut[y] + s * x
                        # lut[y] /= lut[y].norm()
                        ious = F.softmax(lut_ious[y][0:index+1].to(lut_tmp.device), dim=0)
                       # ious = F.softmax((lut_ious[y][0:index + 1]**2).to(lut_tmp.device), dim=0)
                        lut_tmp[y] = (1.0 - ious[-1]) * lut_tmp[y] + ious[-1] * x
                        lut_tmp[y] /= lut_tmp[y].norm()

            else:
                cq[header] = x
                header = (header + 1) % cq.size(0)
        return grad_inputs, None, None, None, None, None, None, None, None, None


def loim(inputs, targets, lut, lut_tmp, lut_ious, cq, header, momentum=0.5, ious=1.0, eps=0.2):
    return LOIM.apply(inputs, targets, lut_tmp, lut, lut_ious, cq, torch.tensor(header), torch.tensor(momentum), ious, torch.tensor(eps))

class LOIMLoss(nn.Module):
    def __init__(self, num_features, num_pids, num_cq_size, oim_momentum, oim_scaler, eps):
        super(LOIMLoss, self).__init__()
        self.num_features = num_features
        self.num_pids = num_pids
        self.num_unlabeled = num_cq_size
        self.momentum = oim_momentum
        self.oim_scaler = oim_scaler
        self.oim_eps = eps

        self.register_buffer("lut", torch.zeros(self.num_pids, self.num_features))
        self.register_buffer("lut_tmp", torch.zeros(self.num_pids, self.num_features))
        self.register_buffer("lut_ious", torch.full((self.num_pids, 5), -1.))
        self.register_buffer("cq", torch.zeros(self.num_unlabeled, self.num_features))

        self.header_cq = 0
    def forward(self, inputs, roi_label, ious):
        targets = torch.cat(roi_label)
        label = targets - 1   #background = -1

        inds = label >= 0
        label = label[inds]
        ious = ious[inds]
        inputs = inputs[inds.unsqueeze(1).expand_as(inputs)].view(-1, self.num_features)

        projected = loim(inputs, label, self.lut, self.lut_tmp, self.lut_ious, self.cq, self.header_cq, momentum=self.momentum, ious=ious, eps=self.oim_eps)
        projected *= self.oim_scaler

        self.header_cq = (
            self.header_cq + (label >= self.num_pids).long().sum().item()
        ) % self.num_unlabeled
        loss_oim = F.cross_entropy(projected, label, ignore_index=5554)
        loss_oim = torch.nan_to_num(loss_oim)
        return loss_oim








