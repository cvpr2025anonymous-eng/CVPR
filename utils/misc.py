
import os
import random
import subprocess
import time
from collections import OrderedDict, defaultdict, deque
import datetime
import pickle
from typing import Optional, List
from skimage import measure
import json, time
import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor

import colorsys
import torch.nn.functional as F

import cv2
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn.functional as F
from skimage import measure


import torchvision


class SmoothedValue(object):


    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):

        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        if d.shape[0] == 0:
            return 0
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):

    world_size = get_world_size()
    if world_size == 1:
        return [data]


    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")


    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)




    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):

    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []

        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict



def masks_to_boxes(masks):

    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)
    y = y.to(masks)
    x = x.to(masks)

    x_mask = ((masks > 128) * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks > 128), 1e8).flatten(1).min(-1)[0]

    y_mask = ((masks > 128) * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks > 128), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_noise(boxes, box_noise_scale=0):
    known_bbox_expand = box_xyxy_to_cxcywh(boxes)

    diff = torch.zeros_like(known_bbox_expand)
    diff[:, :2] = known_bbox_expand[:, 2:] / 2
    diff[:, 2:] = known_bbox_expand[:, 2:]
    known_bbox_expand += torch.mul((torch.rand_like(known_bbox_expand) * 2 - 1.0), diff).cuda() * box_noise_scale
    boxes = box_cxcywh_to_xyxy(known_bbox_expand)
    boxes = boxes.clamp(min=0.0, max=1024)

    return boxes


def masks_sample_points(masks, k=10):

    if masks.numel() == 0:
        return torch.zeros((0, 2), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)
    y = y.to(masks)
    x = x.to(masks)


    samples = []
    for b_i in range(len(masks)):
        select_mask = (masks[b_i] > 128)
        x_idx = torch.masked_select(x, select_mask)
        y_idx = torch.masked_select(y, select_mask)

        perm = torch.randperm(x_idx.size(0))
        idx = perm[:k]
        samples_x = x_idx[idx]
        samples_y = y_idx[idx]
        samples_xy = torch.cat((samples_x[:, None], samples_y[:, None]), dim=1)
        samples.append(samples_xy)

    samples = torch.stack(samples)
    return samples




def masks_noise(masks):
    def get_incoherent_mask(input_masks, sfact):
        mask = input_masks.float()
        w = input_masks.shape[-1]
        h = input_masks.shape[-2]
        mask_small = F.interpolate(mask, (h // sfact, w // sfact), mode='bilinear')
        mask_recover = F.interpolate(mask_small, (h, w), mode='bilinear')
        mask_residue = (mask - mask_recover).abs()
        mask_residue = (mask_residue >= 0.01).float()
        return mask_residue

    gt_masks_vector = masks / 255
    mask_noise = torch.randn(gt_masks_vector.shape, device=gt_masks_vector.device) * 1.0
    inc_masks = get_incoherent_mask(gt_masks_vector, 8)
    gt_masks_vector = ((gt_masks_vector + mask_noise * inc_masks) > 0.5).float()
    gt_masks_vector = gt_masks_vector * 255

    return gt_masks_vector





def _ensure_nchw(x: torch.Tensor, name: str = "tensor") -> torch.Tensor:

    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x)}")

    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        x = x.unsqueeze(1)
    elif x.dim() == 4:
        pass
    else:
        raise ValueError(f"{name} must have 2/3/4 dims, but got shape {tuple(x.shape)}")

    return x.float()


def _binarize_target(target: torch.Tensor) -> torch.Tensor:

    target = _ensure_nchw(target, "target")
    return (target > 0).float()


def _to_probability(pred: torch.Tensor, from_logits=None) -> torch.Tensor:

    pred = _ensure_nchw(pred, "pred")

    if from_logits is None:
        pred_min = float(pred.detach().min())
        pred_max = float(pred.detach().max())
        from_logits = (pred_min < 0.0) or (pred_max > 1.0)

    if from_logits:
        prob = torch.sigmoid(pred)
    else:
        prob = pred.clamp(0.0, 1.0)

    return prob


def _binarize_prediction(
    pred: torch.Tensor,
    threshold: float = 0.5,
    from_logits=None,
) -> tuple[torch.Tensor, torch.Tensor]:

    prob = _to_probability(pred, from_logits=from_logits)
    pred_bin = (prob > threshold).float()
    return pred_bin, prob

class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():


            if meter.count > 0:
                loss_str.append(
                    "{}: {}".format(name, str(meter))
                )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, logger=None):
        if logger is None:
            print_func = print
        else:
            print_func = logger.info

        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj

            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print_func(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print_func(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print_func('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()

    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def setup_for_distributed(is_master):

    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def init_distributed_mode(args):
    if 'WORLD_SIZE' in os.environ and os.environ['WORLD_SIZE'] != '':










        local_world_size = int(os.environ['WORLD_SIZE'])
        args.world_size = args.world_size * local_world_size
        args.gpu = args.local_rank = int(os.environ['LOCAL_RANK'])
        args.rank = args.rank * local_world_size + args.local_rank
        print('world size: {}, rank: {}, local rank: {}'.format(args.world_size, args.rank, args.local_rank))
        print(json.dumps(dict(os.environ), indent=2))
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.local_rank = int(os.environ['SLURM_LOCALID'])
        args.world_size = int(os.environ['SLURM_NPROCS'])

        print('world size: {}, world rank: {}, local rank: {}, device_count: {}'.format(args.world_size, args.rank,
                                                                                        args.local_rank,
                                                                                        torch.cuda.device_count()))
    else:
        print('Not using distributed mode')
        args.distributed = False
        args.world_size = 1
        args.rank = 0
        args.local_rank = 0
        return

    print("world_size:{} rank:{} local_rank:{}".format(args.world_size, args.rank, args.local_rank))
    args.distributed = True
    torch.cuda.set_device(args.local_rank)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    print("Before torch.distributed.barrier()")
    torch.distributed.barrier()
    print("End torch.distributed.barrier()")
    setup_for_distributed(args.rank == 0)





def mask_iou(
    pred_label: torch.Tensor,
    label: torch.Tensor,
    threshold: float = 0.5,
    from_logits=None,
    eps: float = 1e-6,
) -> torch.Tensor:

    pred_bin, _ = _binarize_prediction(pred_label, threshold=threshold, from_logits=from_logits)
    gt_bin = _binarize_target(label)

    if pred_bin.shape[-2:] != gt_bin.shape[-2:]:
        pred_bin = F.interpolate(pred_bin, size=gt_bin.shape[-2:], mode="nearest")

    intersection = (pred_bin * gt_bin).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + gt_bin.sum(dim=(1, 2, 3)) - intersection


    iou = torch.where(union > 0, intersection / (union + eps), torch.ones_like(union))
    return iou.mean().float()





def mask_to_boundary(mask: np.ndarray, dilation_ratio: float = 0.02) -> np.ndarray:

    if mask.ndim != 2:
        raise ValueError(f"mask_to_boundary expects 2D mask [H,W], got {mask.shape}")

    mask = (mask > 0).astype(np.uint8)

    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = max(1, int(round(dilation_ratio * img_diag)))

    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(padded, kernel, iterations=dilation)
    eroded = eroded[1:h + 1, 1:w + 1]

    boundary = mask - eroded
    return (boundary > 0).astype(np.uint8)


def boundary_iou(
    gt: torch.Tensor,
    dt: torch.Tensor,
    dilation_ratio: float = 0.02,
    threshold: float = 0.5,
    from_logits=None,
) -> torch.Tensor:

    gt_bin = _binarize_target(gt)
    dt_bin, _ = _binarize_prediction(dt, threshold=threshold, from_logits=from_logits)

    if dt_bin.shape[-2:] != gt_bin.shape[-2:]:
        dt_bin = F.interpolate(dt_bin, size=gt_bin.shape[-2:], mode="nearest")

    gt_np = gt_bin.squeeze(1).detach().cpu().numpy().astype(np.uint8)
    dt_np = dt_bin.squeeze(1).detach().cpu().numpy().astype(np.uint8)

    ious = []
    for i in range(gt_np.shape[0]):
        gt_boundary = mask_to_boundary(gt_np[i], dilation_ratio)
        dt_boundary = mask_to_boundary(dt_np[i], dilation_ratio)

        intersection = np.logical_and(gt_boundary > 0, dt_boundary > 0).sum()
        union = np.logical_or(gt_boundary > 0, dt_boundary > 0).sum()

        if union == 0:
            ious.append(1.0)
        else:
            ious.append(intersection / union)

    device = gt_bin.device
    return torch.tensor(np.mean(ious), dtype=torch.float32, device=device)





class PD_FA:


    def __init__(self, nclass=1, bins=10, match_distance=3, from_logits=None):
        super(PD_FA, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.match_distance = match_distance
        self.from_logits = from_logits
        self.reset()

    def update(self, preds: torch.Tensor, labels: torch.Tensor):

        labels_bin = _binarize_target(labels)
        preds_prob = _to_probability(preds, from_logits=self.from_logits)

        if preds_prob.shape[-2:] != labels_bin.shape[-2:]:
            preds_prob = F.interpolate(
                preds_prob, size=labels_bin.shape[-2:], mode="bilinear", align_corners=False
            )

        preds_np = (preds_prob.squeeze(1).detach().cpu().numpy() * 255.0).astype(np.float32)
        labels_np = labels_bin.squeeze(1).detach().cpu().numpy().astype(np.uint8)

        B = preds_np.shape[0]

        for b in range(B):
            pred_map = preds_np[b]
            label_map = labels_np[b]
            H, W = pred_map.shape

            self.total_pixels += H * W

            gt_cc = measure.label(label_map, connectivity=2)
            gt_props = measure.regionprops(gt_cc)

            for i_bin in range(self.bins + 1):
                score_thresh = i_bin * (255.0 / self.bins)
                pred_bin = (pred_map > score_thresh).astype(np.uint8)

                pred_cc = measure.label(pred_bin, connectivity=2)
                pred_props = measure.regionprops(pred_cc)

                self.target[i_bin] += len(gt_props)

                pred_areas = [prop.area for prop in pred_props]
                matched_pred_idx = set()
                matched_count = 0

                for gt_prop in gt_props:
                    centroid_gt = np.array(gt_prop.centroid)

                    best_idx = -1
                    best_dist = 1e9

                    for m, pred_prop in enumerate(pred_props):
                        if m in matched_pred_idx:
                            continue

                        centroid_pred = np.array(pred_prop.centroid)
                        distance = np.linalg.norm(centroid_pred - centroid_gt)

                        if distance < self.match_distance and distance < best_dist:
                            best_dist = distance
                            best_idx = m

                    if best_idx != -1:
                        matched_pred_idx.add(best_idx)
                        matched_count += 1

                unmatched_areas = [
                    area for idx, area in enumerate(pred_areas)
                    if idx not in matched_pred_idx
                ]

                self.FA[i_bin] += np.sum(unmatched_areas)
                self.PD[i_bin] += matched_count

    def get(self):
        if self.total_pixels == 0:
            final_fa = np.zeros(self.bins + 1, dtype=np.float32)
        else:
            final_fa = self.FA / float(self.total_pixels)

        final_pd = np.divide(
            self.PD,
            self.target,
            out=np.zeros_like(self.PD, dtype=np.float32),
            where=self.target > 0,
        )

        return final_fa.astype(np.float32), final_pd.astype(np.float32)

    def reset(self):
        self.FA = np.zeros(self.bins + 1, dtype=np.float64)
        self.PD = np.zeros(self.bins + 1, dtype=np.float64)
        self.target = np.zeros(self.bins + 1, dtype=np.float64)
        self.total_pixels = 0





class mIoU:


    def __init__(self, nclass=1, threshold=0.5, from_logits=None):
        super(mIoU, self).__init__()
        self.nclass = nclass
        self.threshold = threshold
        self.from_logits = from_logits
        self.reset()

    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        correct, labeled = batch_pix_accuracy(
            preds, labels,
            threshold=self.threshold,
            from_logits=self.from_logits,
        )
        inter, union = batch_intersection_union(
            preds, labels,
            threshold=self.threshold,
            from_logits=self.from_logits,
        )

        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def get(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU_value = IoU.mean()
        return pixAcc, mIoU_value

    def reset(self):
        self.total_inter = np.zeros(1, dtype=np.float64)
        self.total_union = np.zeros(1, dtype=np.float64)
        self.total_correct = 0.0
        self.total_label = 0.0





def batch_pix_accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    from_logits=None,
):

    pred_bin, _ = _binarize_prediction(output, threshold=threshold, from_logits=from_logits)
    target_bin = _binarize_target(target)

    if pred_bin.shape[-2:] != target_bin.shape[-2:]:
        pred_bin = F.interpolate(pred_bin, size=target_bin.shape[-2:], mode="nearest")

    assert pred_bin.shape == target_bin.shape, "Predict and Label Shape Don't Match"

    pixel_correct = (pred_bin == target_bin).float().sum().item()
    pixel_labeled = float(target_bin.numel())

    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(
    output: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    from_logits=None,
):

    pred_bin, _ = _binarize_prediction(output, threshold=threshold, from_logits=from_logits)
    target_bin = _binarize_target(target)

    if pred_bin.shape[-2:] != target_bin.shape[-2:]:
        pred_bin = F.interpolate(pred_bin, size=target_bin.shape[-2:], mode="nearest")

    assert pred_bin.shape == target_bin.shape, "Predict and Label Shape Don't Match"

    area_inter = float((pred_bin * target_bin).sum().item())
    area_pred = float(pred_bin.sum().item())
    area_lab = float(target_bin.sum().item())
    area_union = area_pred + area_lab - area_inter

    assert area_inter <= area_union + 1e-6, "Intersection area should be smaller than union area"

    return (
        np.array([area_inter], dtype=np.float64),
        np.array([area_union], dtype=np.float64),
    )
