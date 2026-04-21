from __future__ import print_function, division

import numpy as np
import random
from copy import deepcopy
from skimage import io
import os
from glob import glob
import clip
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, utils
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from .untils import tokenize

DATASET_PIXEL_STATS = {
    "NUAA-SIRST": {"mean": [110.34], "std": [58.02]},
    "NUAA_SIRST": {"mean": [110.34], "std": [58.02]},
    "NUDT-SIRST": {"mean": [107.02], "std": [55.56]},
    "NUDT_SIRST": {"mean": [107.02], "std": [55.56]},
    "XD-SIRST": {"mean": [87.35], "std": [59.71]},
    "XD_SIRST": {"mean": [87.35], "std": [59.71]},
}


def resolve_pixel_stats(train_datasets, valid_datasets):
    dataset_candidates = list(train_datasets) + list(valid_datasets)
    for dataset in dataset_candidates:
        dataset_name = dataset.get("name", "")
        dataset_path = dataset.get("im_dir", "")
        for key, stats in DATASET_PIXEL_STATS.items():
            if key in dataset_name or key in dataset_path:
                return stats["mean"], stats["std"], key

    return None, None, "sam_default"


def get_im_gt_name_dict(datasets, flag='valid'):
    print("------------------------------", flag, "--------------------------------")
    name_im_gt_list = []

    for i in range(len(datasets)):
        print("--->>>", flag, " dataset ", i, "/", len(datasets), " ", datasets[i]["name"], "<<<---")

        tmp_im_list = glob(os.path.join(datasets[i]["im_dir"], '*' + datasets[i]["im_ext"]))
        print('-im-', datasets[i]["name"], datasets[i]["im_dir"], ': ', len(tmp_im_list))

        if datasets[i]["gt_dir"] == "":
            print('-gt-', datasets[i]["name"], datasets[i]["gt_dir"], ': ', 'No Ground Truth Found')
            tmp_gt_list = []
        else:
            tmp_gt_list = [
                datasets[i]["gt_dir"] + os.sep + x.split(os.sep)[-1].split(datasets[i]["im_ext"])[0] + datasets[i]["gt_ext"]
                for x in tmp_im_list
            ]
            print('-gt-', datasets[i]["name"], datasets[i]["gt_dir"], ': ', len(tmp_gt_list))

        if datasets[i]["des_dir"] == "":
            print('-des-', datasets[i]["name"], datasets[i]["des_dir"], ': ', 'No description Found')
            tmp_des_list = []
        else:
            tmp_des_list = [
                datasets[i]["des_dir"] + os.sep + x.split(os.sep)[-1].split(datasets[i]["im_ext"])[0] + datasets[i]["des_ext"]
                for x in tmp_im_list
            ]
            print('-des-', datasets[i]["name"], datasets[i]["des_dir"], ': ', len(tmp_des_list))

        name_im_gt_list.append({
            "dataset_name": datasets[i]["name"],
            "im_path": tmp_im_list,
            "gt_path": tmp_gt_list,
            "des_path": tmp_des_list,
            "im_ext": datasets[i]["im_ext"],
            "gt_ext": datasets[i]["gt_ext"],
            "des_ext": datasets[i]["des_ext"],
        })

    return name_im_gt_list


def valid_collate_fn(batch):
    out = {}

    out["imidx"] = torch.stack([x["imidx"] for x in batch], dim=0)
    out["image"] = torch.stack([x["image"] for x in batch], dim=0)
    out["label"] = torch.stack([x["label"] for x in batch], dim=0)
    out["shape"] = torch.stack([x["shape"] for x in batch], dim=0)

    out["caption"] = [x["caption"] for x in batch]

    if "ori_label" in batch[0]:
        out["ori_label"] = [x["ori_label"] for x in batch]
    if "ori_im_path" in batch[0]:
        out["ori_im_path"] = [x["ori_im_path"] for x in batch]
    if "ori_gt_path" in batch[0]:
        out["ori_gt_path"] = [x["ori_gt_path"] for x in batch]
    if "ori_des_path" in batch[0]:
        out["ori_des_path"] = [x["ori_des_path"] for x in batch]
    return out


def create_dataloaders(name_im_gt_list, my_transforms=[], batch_size=1, training=False):
    gos_dataloaders = []
    gos_datasets = []

    if len(name_im_gt_list) == 0:
        return gos_dataloaders, gos_datasets

    num_workers_ = 1
    if batch_size > 1:
        num_workers_ = 2
    if batch_size > 4:
        num_workers_ = 4
    if batch_size > 8:
        num_workers_ = 8

    if training:
        for i in range(len(name_im_gt_list)):
            gos_dataset = OnlineDataset([name_im_gt_list[i]], transform=transforms.Compose(my_transforms))
            gos_datasets.append(gos_dataset)

        gos_dataset = ConcatDataset(gos_datasets)
        dataloader = DataLoader(
            gos_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers_,
            drop_last=True,
        )
        gos_dataloaders = dataloader
        gos_datasets = gos_dataset
    else:
        for i in range(len(name_im_gt_list)):
            gos_dataset = OnlineDataset(
                [name_im_gt_list[i]],
                transform=transforms.Compose(my_transforms),
                eval_ori_resolution=True,
            )
            dataloader = DataLoader(
                gos_dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers_,
                collate_fn=valid_collate_fn,
            )
            gos_dataloaders.append(dataloader)
            gos_datasets.append(gos_dataset)

    return gos_dataloaders, gos_datasets


class RandomHFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        imidx, image, label, shape, caption = sample['imidx'], sample['image'], sample['label'], sample['shape'], sample['caption']

        if random.random() < self.prob:
            image = torch.flip(image, dims=[2])
            label = torch.flip(label, dims=[2])
        return {'imidx': imidx, 'image': image, 'label': label, 'shape': shape, 'caption': caption}


class RandomVFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        imidx, image, label, shape, caption = sample['imidx'], sample['image'], sample['label'], sample['shape'], sample['caption']

        if random.random() < self.prob:
            image = torch.flip(image, dims=[1])
            label = torch.flip(label, dims=[1])
        return {'imidx': imidx, 'image': image, 'label': label, 'shape': shape, 'caption': caption}


class Resize(object):
    def __init__(self, size=[320, 320]):
        self.size = size

    def __call__(self, sample):
        imidx, image, label, shape, caption = sample['imidx'], sample['image'], sample['label'], sample['shape'], sample['caption']

        image = torch.squeeze(F.interpolate(torch.unsqueeze(image, 0), self.size, mode='bilinear'), dim=0)
        label = torch.squeeze(F.interpolate(torch.unsqueeze(label, 0), self.size, mode='nearest'), dim=0)
        return {'imidx': imidx, 'image': image, 'label': label, 'shape': torch.tensor(self.size), 'caption': caption}


class RandomCrop(object):
    def __init__(self, size=[288, 288]):
        self.size = size

    def __call__(self, sample):
        imidx, image, label, shape, caption = sample['imidx'], sample['image'], sample['label'], sample['shape'], sample['caption']

        h, w = image.shape[1:]
        new_h, new_w = self.size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[:, top:top + new_h, left:left + new_w]
        label = label[:, top:top + new_h, left:left + new_w]
        return {'imidx': imidx, 'image': image, 'label': label, 'shape': torch.tensor(self.size), 'caption': caption}


class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        return sample


class LargeScaleJitter(object):
    def __init__(self, output_size=1024, aug_scale_min=0.1, aug_scale_max=2.0):
        self.desired_size = torch.tensor(output_size)
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max

    def pad_target(self, padding, target):
        target = target.copy()
        if "masks" in target:
            target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[1], 0, padding[0]))
        return target

    def __call__(self, sample):
        imidx, image, label, image_size, caption = sample['imidx'], sample['image'], sample['label'], sample['shape'], sample['caption']

        out_desired_size = (self.desired_size * image_size / max(image_size)).round().int()

        random_scale = torch.rand(1) * (self.aug_scale_max - self.aug_scale_min) + self.aug_scale_min
        scaled_size = (random_scale * self.desired_size).round()

        scale = torch.minimum(scaled_size / image_size[0], scaled_size / image_size[1])
        scaled_size = (image_size * scale).round().long()

        scaled_image = torch.squeeze(
            F.interpolate(torch.unsqueeze(image, 0), scaled_size.tolist(), mode='bilinear'),
            dim=0
        )
        scaled_label = torch.squeeze(
            F.interpolate(torch.unsqueeze(label, 0), scaled_size.tolist(), mode='nearest'),
            dim=0
        )

        crop_size = (min(self.desired_size, scaled_size[0]), min(self.desired_size, scaled_size[1]))

        margin_h = max(scaled_size[0] - crop_size[0], 0).item()
        margin_w = max(scaled_size[1] - crop_size[1], 0).item()
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0].item()
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1].item()

        scaled_image = scaled_image[:, crop_y1:crop_y2, crop_x1:crop_x2]
        scaled_label = scaled_label[:, crop_y1:crop_y2, crop_x1:crop_x2]

        padding_h = max(self.desired_size - scaled_image.size(1), 0).item()
        padding_w = max(self.desired_size - scaled_image.size(2), 0).item()
        image = F.pad(scaled_image, [0, padding_w, 0, padding_h], value=128)
        label = F.pad(scaled_label, [0, padding_w, 0, padding_h], value=0)
        return {'imidx': imidx, 'image': image, 'label': label, 'shape': torch.tensor(image.shape[-2:]), 'caption': caption}


class TargetAwareCropResize(object):
    def __init__(
        self,
        output_size=1024,
        target_frac_min=0.03,
        target_frac_max=0.08,
        global_view_prob=0.25,
        min_crop_size=96,
    ):
        self.output_size = output_size
        self.target_frac_min = target_frac_min
        self.target_frac_max = target_frac_max
        self.global_view_prob = global_view_prob
        self.min_crop_size = min_crop_size

    def _resize_and_pad(self, image, label):
        h, w = image.shape[-2:]
        scale = min(self.output_size / h, self.output_size / w)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))

        image = torch.squeeze(
            F.interpolate(
                torch.unsqueeze(image, 0),
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False,
            ),
            dim=0,
        )
        label = torch.squeeze(
            F.interpolate(
                torch.unsqueeze(label, 0),
                size=(new_h, new_w),
                mode='nearest',
            ),
            dim=0,
        )

        pad_h = self.output_size - new_h
        pad_w = self.output_size - new_w
        image = F.pad(image, [0, pad_w, 0, pad_h], value=128)
        label = F.pad(label, [0, pad_w, 0, pad_h], value=0)
        return image, label

    def __call__(self, sample):
        imidx, image, label, shape, caption = sample['imidx'], sample['image'], sample['label'], sample['shape'], sample['caption']

        h, w = image.shape[-2:]

        if random.random() < self.global_view_prob:
            image, label = self._resize_and_pad(image, label)
            return {'imidx': imidx, 'image': image, 'label': label, 'shape': torch.tensor(image.shape[-2:]), 'caption': caption}

        pos = torch.nonzero(label[0] > 0, as_tuple=False)

        if pos.numel() == 0:
            image, label = self._resize_and_pad(image, label)
            return {'imidx': imidx, 'image': image, 'label': label, 'shape': torch.tensor(image.shape[-2:]), 'caption': caption}

        y_min = int(pos[:, 0].min().item())
        y_max = int(pos[:, 0].max().item())
        x_min = int(pos[:, 1].min().item())
        x_max = int(pos[:, 1].max().item())

        box_h = y_max - y_min + 1
        box_w = x_max - x_min + 1

        target_frac = random.uniform(self.target_frac_min, self.target_frac_max)
        desired_crop = int(round(max(box_h, box_w) / max(target_frac, 1e-6)))

        crop_h = min(max(desired_crop, box_h + 16, self.min_crop_size), h)
        crop_w = min(max(desired_crop, box_w + 16, self.min_crop_size), w)

        x1_low = max(0, x_max - crop_w + 1)
        x1_high = min(x_min, w - crop_w)
        y1_low = max(0, y_max - crop_h + 1)
        y1_high = min(y_min, h - crop_h)

        if x1_low <= x1_high:
            x1 = random.randint(x1_low, x1_high)
        else:
            x1 = min(max(0, x_min), max(0, w - crop_w))

        if y1_low <= y1_high:
            y1 = random.randint(y1_low, y1_high)
        else:
            y1 = min(max(0, y_min), max(0, h - crop_h))

        x2 = x1 + crop_w
        y2 = y1 + crop_h

        image = image[:, y1:y2, x1:x2]
        label = label[:, y1:y2, x1:x2]

        image = torch.squeeze(
            F.interpolate(
                torch.unsqueeze(image, 0),
                size=(self.output_size, self.output_size),
                mode='bilinear',
                align_corners=False,
            ),
            dim=0,
        )
        label = torch.squeeze(
            F.interpolate(
                torch.unsqueeze(label, 0),
                size=(self.output_size, self.output_size),
                mode='nearest',
            ),
            dim=0,
        )
        return {'imidx': imidx, 'image': image, 'label': label, 'shape': torch.tensor(image.shape[-2:]), 'caption': caption}


class HybridTinyTargetAug(object):
    def __init__(
        self,
        output_size=1024,
        aug_scale_min=0.6,
        aug_scale_max=1.6,
        target_crop_prob=0.2,
        target_frac_min=0.03,
        target_frac_max=0.04,
        min_crop_size=224,
        tiny_box_side_frac=0.05,
        tiny_mask_area_frac=0.0015,
    ):
        self.base_transform = LargeScaleJitter(
            output_size=output_size,
            aug_scale_min=aug_scale_min,
            aug_scale_max=aug_scale_max,
        )
        self.target_transform = TargetAwareCropResize(
            output_size=output_size,
            target_frac_min=target_frac_min,
            target_frac_max=target_frac_max,
            global_view_prob=0.0,
            min_crop_size=min_crop_size,
        )
        self.target_crop_prob = target_crop_prob
        self.tiny_box_side_frac = tiny_box_side_frac
        self.tiny_mask_area_frac = tiny_mask_area_frac

    def _is_tiny_target(self, label):
        pos = torch.nonzero(label[0] > 0, as_tuple=False)
        if pos.numel() == 0:
            return False

        h, w = label.shape[-2:]
        y_min = int(pos[:, 0].min().item())
        y_max = int(pos[:, 0].max().item())
        x_min = int(pos[:, 1].min().item())
        x_max = int(pos[:, 1].max().item())

        box_h = y_max - y_min + 1
        box_w = x_max - x_min + 1
        box_side_frac = max(box_h / max(h, 1), box_w / max(w, 1))
        mask_area_frac = label[0].float().mean().item()

        return (
            box_side_frac <= self.tiny_box_side_frac
            and mask_area_frac <= self.tiny_mask_area_frac
        )

    def __call__(self, sample):
        if self.target_crop_prob <= 0:
            return self.base_transform(sample)

        if random.random() >= self.target_crop_prob:
            return self.base_transform(sample)

        if not self._is_tiny_target(sample["label"]):
            return self.base_transform(sample)

        return self.target_transform(sample)


class OnlineDataset(Dataset):
    def __init__(self, name_im_gt_list, transform=None, eval_ori_resolution=False):
        self.transform = transform
        self.dataset = {}

        dt_name_list = []
        im_name_list = []
        im_path_list = []
        gt_path_list = []
        des_path_list = []
        im_ext_list = []
        gt_ext_list = []
        des_ext_list = []

        for i in range(0, len(name_im_gt_list)):
            dt_name_list.extend([name_im_gt_list[i]["dataset_name"] for x in name_im_gt_list[i]["im_path"]])
            im_name_list.extend(
                [x.split(os.sep)[-1].split(name_im_gt_list[i]["im_ext"])[0] for x in name_im_gt_list[i]["im_path"]]
            )
            im_path_list.extend(name_im_gt_list[i]["im_path"])
            gt_path_list.extend(name_im_gt_list[i]["gt_path"])
            des_path_list.extend(name_im_gt_list[i]["des_path"])
            im_ext_list.extend([name_im_gt_list[i]["im_ext"] for x in name_im_gt_list[i]["im_path"]])
            gt_ext_list.extend([name_im_gt_list[i]["gt_ext"] for x in name_im_gt_list[i]["gt_path"]])
            des_ext_list.extend([name_im_gt_list[i]["des_ext"] for x in name_im_gt_list[i]["des_path"]])

        self.dataset["data_name"] = dt_name_list
        self.dataset["im_name"] = im_name_list
        self.dataset["im_path"] = im_path_list
        self.dataset["ori_im_path"] = deepcopy(im_path_list)
        self.dataset["gt_path"] = gt_path_list
        self.dataset["ori_gt_path"] = deepcopy(gt_path_list)
        self.dataset["des_path"] = des_path_list
        self.dataset["ori_des_path"] = deepcopy(des_path_list)
        self.dataset["im_ext"] = im_ext_list
        self.dataset["gt_ext"] = gt_ext_list
        self.dataset["des_ext"] = des_ext_list

        self.eval_ori_resolution = eval_ori_resolution

    def __len__(self):
        return len(self.dataset["im_path"])

    def __getitem__(self, idx):
        im_path = self.dataset["im_path"][idx]
        gt_path = self.dataset["gt_path"][idx]
        des_path = self.dataset["des_path"][idx]

        im = io.imread(im_path)
        gt = io.imread(gt_path)
        with open(des_path, 'r') as cf:
            caption = cf.read().strip()
        caption = tokenize(caption)

        if len(gt.shape) > 2:
            gt = gt[:, :, 0]
        if len(im.shape) < 3:
            im = im[:, :, np.newaxis]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)

        im = torch.tensor(im.copy(), dtype=torch.float32)
        im = torch.transpose(torch.transpose(im, 1, 2), 0, 1)

        gt = torch.unsqueeze(torch.tensor(gt, dtype=torch.float32), 0)
        gt = (gt > 0).float()

        sample = {
            "imidx": torch.from_numpy(np.array(idx)),
            "image": im,
            "label": gt,
            "shape": torch.tensor(im.shape[-2:]),
            "caption": caption,
        }

        if self.transform:
            sample = self.transform(sample)

        if self.eval_ori_resolution:
            sample["ori_label"] = gt.type(torch.uint8)
            sample['ori_im_path'] = self.dataset["im_path"][idx]
            sample['ori_gt_path'] = self.dataset["gt_path"][idx]
            sample['ori_des_path'] = self.dataset["des_path"][idx]

        return sample
