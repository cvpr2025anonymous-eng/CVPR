import os
import math
import argparse
import random
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from segment_anything_training.build_sam import sam_model_registry
from lora import LoRA_sam, LoRA_CLIP_Text
from utils.dataloader import (
    get_im_gt_name_dict,
    create_dataloaders,
    HybridTinyTargetAug,
    RandomHFlip,
    RandomVFlip,
    Resize,
    resolve_pixel_stats,
)
from utils.loss_mask import final_iou_loss
import utils.misc as misc


def get_args_parser():
    parser = argparse.ArgumentParser("SAIST", add_help=False)

    parser.add_argument(
        "--output",
        type=str,
        default="work_dirs/saist_exp3",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="vit_b",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="pretrained_checkpoint/sam_vit_b_01ec64.pth",
    )
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--lr_decoder", default=None, type=float)
    parser.add_argument("--lr_prompt", default=None, type=float)
    parser.add_argument("--lr_image_encoder_lora", default=None, type=float)
    parser.add_argument("--lr_image_encoder_neck", default=None, type=float)
    parser.add_argument("--lr_clip_text_lora", default=None, type=float)

    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--warmup_epochs", default=5, type=int)
    parser.add_argument("--warmup_start_factor", default=0.1, type=float)
    parser.add_argument("--min_lr", default=1e-6, type=float)

    parser.add_argument("--start_epoch", default=1, type=int)
    parser.add_argument("--max_epoch_num", default=200, type=int)
    parser.add_argument(
        "--input_size",
        nargs=2,
        type=int,
        default=[1024, 1024],
    )

    parser.add_argument("--batch_size_train", default=8, type=int)
    parser.add_argument("--batch_size_valid", default=8, type=int)

    parser.add_argument("--model_save_fre", default=50, type=int)

    parser.add_argument("--sam_lora_rank", default=2, type=int)
    parser.add_argument("--sam_lora_alpha", default=4.0, type=float)
    parser.add_argument("--clip_lora_rank", default=2, type=int)
    parser.add_argument("--clip_lora_alpha", default=4.0, type=float)
    parser.add_argument("--clip_lora_dropout", default=0.0, type=float)

    parser.add_argument("--contrastive_weight", default=0.5, type=float)
    parser.add_argument("--contrastive_lambda", default=0.1, type=float)
    parser.add_argument(
        "--mmd_lambda",
        default=0.05,
        type=float,
    )
    parser.add_argument(
        "--contrastive_temperature",
        default=0.07,
        type=float,
    )
    parser.add_argument(
        "--sam_mask_weight",
        default=0.0,
        type=float,
    )

    parser.add_argument(
        "--scene_source",
        default="image",
        choices=["image", "background"],
    )

    parser.add_argument("--iou_smooth", default=1e-6, type=float)
    parser.add_argument(
        "--supervision_size",
        default=1024,
        type=int,
    )

    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--find_unused_params", action="store_true")

    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--restore-model", type=str, default="")
    parser.add_argument("--eval_interval", default=5, type=int)

    args = parser.parse_args()
    return args


def move_caption_to_device(captions, device):
    if torch.is_tensor(captions):
        return captions.to(device, non_blocking=True)

    if isinstance(captions, list):
        out = []
        for c in captions:
            if torch.is_tensor(c):
                out.append(c.to(device, non_blocking=True))
            else:
                out.append(c)
        return out

    return captions


def compute_contrastive_loss(text_tokens, image_tokens, temperature=0.07):
    text_tokens = F.normalize(text_tokens, dim=-1)
    image_tokens = F.normalize(image_tokens, dim=-1)

    logits_per_image = torch.matmul(image_tokens, text_tokens.t()) / temperature
    logits_per_text = logits_per_image.t()

    labels = torch.arange(text_tokens.size(0), device=text_tokens.device)

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)

    return 0.5 * (loss_i + loss_t)


def compute_mmd_loss(
    text_tokens,
    image_tokens,
    bandwidths=(0.5, 1.0, 2.0, 4.0, 8.0),
):
    text_tokens = F.normalize(text_tokens, dim=-1)
    image_tokens = F.normalize(image_tokens, dim=-1)

    dist_xx = torch.cdist(text_tokens, text_tokens, p=2).pow(2)
    dist_yy = torch.cdist(image_tokens, image_tokens, p=2).pow(2)
    dist_xy = torch.cdist(text_tokens, image_tokens, p=2).pow(2)

    bandwidths = text_tokens.new_tensor(bandwidths).view(-1, 1, 1)
    kernel_xx = torch.exp(-dist_xx.unsqueeze(0) / (2.0 * bandwidths)).mean(dim=0)
    kernel_yy = torch.exp(-dist_yy.unsqueeze(0) / (2.0 * bandwidths)).mean(dim=0)
    kernel_xy = torch.exp(-dist_xy.unsqueeze(0) / (2.0 * bandwidths)).mean(dim=0)

    return kernel_xx.mean() + kernel_yy.mean() - 2.0 * kernel_xy.mean()


def compute_alignment_losses(text_tokens, image_tokens, args, device):
    infonce_loss = torch.zeros((), device=device)
    mmd_loss = torch.zeros((), device=device)

    if (
        text_tokens is None
        or image_tokens is None
        or not torch.is_tensor(text_tokens)
        or not torch.is_tensor(image_tokens)
        or text_tokens.dim() != 2
        or image_tokens.dim() != 2
        or text_tokens.size(0) <= 1
        or image_tokens.size(0) <= 1
    ):
        return infonce_loss, mmd_loss, infonce_loss + mmd_loss

    if args.contrastive_lambda > 0:
        infonce_loss = args.contrastive_lambda * compute_contrastive_loss(
            text_tokens=text_tokens,
            image_tokens=image_tokens,
            temperature=args.contrastive_temperature,
        )

    if args.mmd_lambda > 0:
        mmd_loss = args.mmd_lambda * compute_mmd_loss(
            text_tokens=text_tokens,
            image_tokens=image_tokens,
        )

    return infonce_loss, mmd_loss, infonce_loss + mmd_loss


def unpack_model_outputs(outputs):
    pred_masks = outputs
    text_tokens = None
    image_tokens = None
    aux_outputs = {}

    if isinstance(outputs, (tuple, list)):
        if len(outputs) == 4:
            pred_masks, text_tokens, image_tokens, aux_outputs = outputs

        elif len(outputs) == 3:
            pred_masks, second, third = outputs
            if isinstance(third, dict):
                aux_outputs = third
            else:
                text_tokens = second
                image_tokens = third

        elif len(outputs) == 2:
            pred_masks, second = outputs
            if isinstance(second, dict):
                aux_outputs = second

        elif len(outputs) == 1:
            pred_masks = outputs[0]

        else:
            raise ValueError(f"Unsupported model outputs length: {len(outputs)}")

    if aux_outputs is None:
        aux_outputs = {}

    return pred_masks, text_tokens, image_tokens, aux_outputs


def is_lora_parameter_name(name: str) -> bool:
    return "lora_" in name


def set_trainable_by_policy(model):
    for _, param in model.named_parameters():
        param.requires_grad = False

    prompt_encoder_trainable_prefixes = (
        "prompt_encoder.mask_prompt_",
        "prompt_encoder.no_mask_embed",
        "prompt_encoder.sr_clip.text_adapter",
        "prompt_encoder.sr_clip.visual_adapter",
        "prompt_encoder.sr_clip.clip_backbone.scene_prompt_proj",
        "prompt_encoder.sr_clip.clip_backbone.scene_token_expander",
        "prompt_encoder.sr_clip.clip_backbone.text_token_expander",
        "prompt_encoder.sr_clip.clip_backbone.object_object_refine",
        "prompt_encoder.sr_clip.clip_backbone.scene_object_refine",
    )

    for name, param in model.named_parameters():
        if name.startswith("mask_decoder"):
            param.requires_grad = True

        if name.startswith(prompt_encoder_trainable_prefixes):
            param.requires_grad = True

        if "prompt_encoder.sr_clip.clip_backbone.transformer" in name and is_lora_parameter_name(name):
            param.requires_grad = True

        if name.startswith("image_encoder") and is_lora_parameter_name(name):
            param.requires_grad = True

        if name.startswith("image_encoder.neck"):
            param.requires_grad = True


def resolve_group_lrs(args):
    lr_decoder = args.lr_decoder if args.lr_decoder is not None else args.learning_rate
    lr_prompt = args.lr_prompt if args.lr_prompt is not None else args.learning_rate
    lr_image_encoder_lora = (
        args.lr_image_encoder_lora
        if args.lr_image_encoder_lora is not None
        else args.learning_rate * 0.3
    )
    lr_image_encoder_neck = (
        args.lr_image_encoder_neck
        if args.lr_image_encoder_neck is not None
        else args.learning_rate * 0.5
    )
    lr_clip_text_lora = (
        args.lr_clip_text_lora
        if args.lr_clip_text_lora is not None
        else args.learning_rate * 0.5
    )

    return {
        "decoder": lr_decoder,
        "prompt": lr_prompt,
        "image_encoder_lora": lr_image_encoder_lora,
        "image_encoder_neck": lr_image_encoder_neck,
        "clip_text_lora": lr_clip_text_lora,
    }


def build_optimizer(args, sam):
    lr_dict = resolve_group_lrs(args)

    prompt_encoder_trainable_prefixes = (
        "prompt_encoder.mask_prompt_",
        "prompt_encoder.no_mask_embed",
        "prompt_encoder.sr_clip.text_adapter",
        "prompt_encoder.sr_clip.visual_adapter",
        "prompt_encoder.sr_clip.clip_backbone.scene_prompt_proj",
        "prompt_encoder.sr_clip.clip_backbone.scene_token_expander",
        "prompt_encoder.sr_clip.clip_backbone.text_token_expander",
        "prompt_encoder.sr_clip.clip_backbone.object_object_refine",
        "prompt_encoder.sr_clip.clip_backbone.scene_object_refine",
    )

    param_groups = {
        "decoder": [],
        "prompt": [],
        "image_encoder_lora": [],
        "image_encoder_neck": [],
        "clip_text_lora": [],
    }

    for name, param in sam.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith("mask_decoder"):
            param_groups["decoder"].append(param)

        elif name.startswith("image_encoder.neck"):
            param_groups["image_encoder_neck"].append(param)

        elif name.startswith("image_encoder") and is_lora_parameter_name(name):
            param_groups["image_encoder_lora"].append(param)

        elif (
            "prompt_encoder.sr_clip.clip_backbone.transformer" in name
            and is_lora_parameter_name(name)
        ):
            param_groups["clip_text_lora"].append(param)

        elif name.startswith(prompt_encoder_trainable_prefixes):
            param_groups["prompt"].append(param)

        else:
            raise RuntimeError(
                f"Trainable parameter not assigned to any LR group: {name}"
            )

    optimizer_grouped_parameters = []
    for group_name, params in param_groups.items():
        if len(params) == 0:
            continue
        base_lr = lr_dict[group_name]
        optimizer_grouped_parameters.append(
            {
                "name": group_name,
                "params": params,
                "lr": base_lr,
                "initial_lr": base_lr,
            }
        )

    if len(optimizer_grouped_parameters) == 0:
        raise RuntimeError("No trainable parameters found for optimizer.")

    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.weight_decay,
    )

    return optimizer


def get_lr_for_epoch(base_lr, epoch, args):
    warmup_epochs = max(int(args.warmup_epochs), 0)
    min_lr = float(args.min_lr)
    max_epoch_num = max(int(args.max_epoch_num), 1)

    if warmup_epochs > 0 and epoch <= warmup_epochs:
        alpha = epoch / float(warmup_epochs)
        factor = args.warmup_start_factor + (1.0 - args.warmup_start_factor) * alpha
        return max(min_lr, base_lr * factor)

    if max_epoch_num <= warmup_epochs:
        return max(min_lr, base_lr)

    progress = (epoch - warmup_epochs) / float(max(1, max_epoch_num - warmup_epochs))
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def set_optimizer_lrs(optimizer, epoch, args):
    for group in optimizer.param_groups:
        base_lr = group.get("initial_lr", group["lr"])
        group["lr"] = get_lr_for_epoch(base_lr, epoch, args)


def get_optimizer_lr_dict(optimizer):
    lr_dict = {}
    for i, group in enumerate(optimizer.param_groups):
        name = group.get("name", f"group_{i}")
        lr_dict[name] = group["lr"]
    return lr_dict


def build_scenes(inputs, labels, scene_source="image"):
    if scene_source == "background":
        labels_bin = (labels > 0.5).float()
        return inputs * (1.0 - labels_bin)
    return inputs


def prepare_batched_input(
    inputs,
    labels,
    captions,
    scene_source="image",
):
    scenes = build_scenes(inputs, labels, scene_source=scene_source)

    batched_input = []
    batch_size = inputs.shape[0]

    for b_i in range(batch_size):
        item = dict()
        item["image"] = inputs[b_i].contiguous().float()
        item["background"] = scenes[b_i].contiguous().float()
        item["caption"] = captions[b_i]
        item["original_size"] = tuple(inputs[b_i].shape[-2:])
        batched_input.append(item)

    return batched_input


def resize_to_label_size(mask_logits, labels):
    if mask_logits.shape[-2:] != labels.shape[-2:]:
        mask_logits = F.interpolate(
            mask_logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    return mask_logits


def resize_labels_for_supervision(labels, args):
    target_size = int(getattr(args, "supervision_size", 0))
    if target_size <= 0:
        return labels
    if labels.shape[-2:] != (target_size, target_size):
        labels = F.interpolate(
            labels.float(),
            size=(target_size, target_size),
            mode="nearest",
        )
    return labels


DATASET_EVAL_SIZES = {
    "NUAA-SIRST": 1024,
    "NUAA_SIRST": 1024,
    "XD-SIRST": 512,
    "XD_SIRST": 512,
    "NUDT-SIRST": 256,
    "NUDT_SIRST": 256,
}


def resolve_eval_size_from_loader(valid_dataloader, default_size=1024):
    dataset = getattr(valid_dataloader, "dataset", None)
    data_names = getattr(dataset, "dataset", {}).get("data_name", []) if dataset is not None else []
    dataset_name = data_names[0] if data_names else ""
    for key, size in DATASET_EVAL_SIZES.items():
        if key in dataset_name:
            return size
    return default_size


def resize_labels_for_metric(labels, target_size):
    if target_size <= 0:
        return labels
    if labels.shape[-2:] != (target_size, target_size):
        labels = F.interpolate(
            labels.float(),
            size=(target_size, target_size),
            mode="nearest",
        )
    return labels


def resize_labels_for_pd_fa(labels, target_size=256):
    if labels.shape[-2:] != (target_size, target_size):
        labels = F.interpolate(
            labels.float(),
            size=(target_size, target_size),
            mode="nearest",
        )
    return labels


def compute_mask_loss(mask_logits, labels, args):
    mask_logits = resize_to_label_size(mask_logits, labels)
    loss_dict = final_iou_loss(
        final_masks=mask_logits,
        targets=labels,
        smooth=args.iou_smooth,
    )
    return mask_logits, loss_dict["loss_total"]


def compute_sam_aux_loss(aux_outputs, labels, args, device):
    if not isinstance(aux_outputs, dict):
        return torch.zeros((), device=device)

    sam_mask_logits = aux_outputs.get("sam_mask", None)
    if sam_mask_logits is None or not torch.is_tensor(sam_mask_logits):
        return torch.zeros((), device=device)

    _, sam_aux_loss = compute_mask_loss(sam_mask_logits, labels, args)
    return sam_aux_loss


def main(train_datasets, valid_datasets, args):
    misc.init_distributed_mode(args)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if args.input_size[0] != args.input_size[1]:
        raise ValueError(
            f"HybridTinyTargetAug currently expects a square scalar output_size, "
            f"but got input_size={args.input_size}"
        )
    train_output_size = int(args.input_size[0])

    if not args.eval:
        print("--- create training dataloader ---")
        train_im_gt_list = get_im_gt_name_dict(train_datasets, flag="train")
        train_dataloaders, _ = create_dataloaders(
            train_im_gt_list,
            my_transforms=[
                HybridTinyTargetAug(
                    output_size=train_output_size,
                ),
                RandomHFlip(prob=0.25),
                RandomVFlip(prob=0.25),
            ],
            batch_size=args.batch_size_train,
            training=True,
        )
        print(len(train_dataloaders), " train dataloaders created")

    print("--- create valid dataloader ---")
    valid_im_gt_list = get_im_gt_name_dict(valid_datasets, flag="valid")
    valid_dataloaders, _ = create_dataloaders(
        valid_im_gt_list,
        my_transforms=[Resize(args.input_size)],
        batch_size=args.batch_size_valid,
        training=False,
    )
    print(len(valid_dataloaders), " valid dataloaders created")

    pixel_mean, pixel_std, _ = resolve_pixel_stats(
        train_datasets=train_datasets,
        valid_datasets=valid_datasets,
    )

    sam = sam_model_registry[args.model_type](
        checkpoint=args.checkpoint,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )

    sam_lora = LoRA_sam(
        sam_model=sam,
        rank=args.sam_lora_rank,
        alpha=args.sam_lora_alpha,
    )
    sam = sam_lora.sam

    _ = LoRA_CLIP_Text(
        clip_backbone=sam.prompt_encoder.sr_clip.clip_backbone,
        rank=args.clip_lora_rank,
        alpha=args.clip_lora_alpha,
        dropout=args.clip_lora_dropout,
    )

    set_trainable_by_policy(sam)
    sam.to(device=args.device)

    if not args.eval:
        print("--- define optimizer ---")
        optimizer = build_optimizer(args, sam)
        train(args, sam, optimizer, train_dataloaders, valid_dataloaders)
    else:
        if args.restore_model:
            print("restore model from:", args.restore_model)
            sam.load_state_dict(
                torch.load(
                    args.restore_model,
                    map_location=args.device,
                )
            )
        evaluate(args, sam, valid_dataloaders, epoch=args.start_epoch)


def train(args, sam, optimizer, train_dataloaders, valid_dataloaders):
    if misc.is_main_process():
        os.makedirs(args.output, exist_ok=True)

    epoch_start = args.start_epoch
    epoch_num = args.max_epoch_num

    sam.train()

    for epoch in range(epoch_start, epoch_num + 1):
        set_optimizer_lrs(optimizer, epoch, args)
        current_lr_dict = get_optimizer_lr_dict(optimizer)

        print(
            "epoch:",
            epoch,
            " lrs:",
            {k: round(v, 8) for k, v in current_lr_dict.items()},
        )

        metric_logger = misc.MetricLogger(delimiter="  ")
        current_contrastive_weight = args.contrastive_weight if epoch < 50 else 0.0

        for data in train_dataloaders:
            inputs, labels, captions = data["image"], data["label"], data["caption"]

            inputs = inputs.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            captions = move_caption_to_device(captions, args.device)

            if labels.dim() == 3:
                labels = labels.unsqueeze(1)
            labels = (labels > 0).float()
            labels_sup = resize_labels_for_supervision(labels, args)

            batched_input = prepare_batched_input(
                inputs,
                labels,
                captions,
                scene_source=args.scene_source,
            )

            outputs = sam(
                batched_input,
                multimask_output=False,
            )
            pred_masks, text_tokens, image_tokens, aux_outputs = unpack_model_outputs(outputs)

            final_masks = pred_masks
            final_masks, final_loss = compute_mask_loss(
                final_masks,
                labels_sup,
                args,
            )

            sam_aux_loss = compute_sam_aux_loss(
                aux_outputs=aux_outputs,
                labels=labels_sup,
                args=args,
                device=final_masks.device,
            )
            mask_loss = final_loss + args.sam_mask_weight * sam_aux_loss

            _, _, contrast_loss = compute_alignment_losses(
                text_tokens=text_tokens,
                image_tokens=image_tokens,
                args=args,
                device=final_masks.device,
            )

            loss = mask_loss + current_contrastive_weight * contrast_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, sam.parameters()),
                max_norm=1.0,
            )

            optimizer.step()

            metric_update = dict(
                loss=loss.item(),
                final_loss=final_loss.item(),
                mask_loss=mask_loss.item(),
                sam_aux_loss=sam_aux_loss.item(),
                contrast_loss=contrast_loss.item(),
            )
            metric_logger.update(**metric_update)

        print("Finished epoch:", epoch)
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)

        if misc.is_main_process():
            with open(os.path.join(args.output, "train.txt"), "a") as f:
                f.write("Finished epoch: {}\n".format(epoch))
                f.write("Averaged stats:\n")
                for key, meter in metric_logger.meters.items():
                    f.write(f"{key}: {meter.global_avg}\n")
                f.write("LRs:\n")
                for k, v in current_lr_dict.items():
                    f.write(f"{k}: {v}\n")
                f.write("\n")

        if epoch == epoch_start or epoch % args.eval_interval == 0:
            test_stats = evaluate(args, sam, valid_dataloaders, epoch=epoch)

            train_stats = {
                k: meter.global_avg
                for k, meter in metric_logger.meters.items()
                if meter.count > 0
            }
            train_stats.update(test_stats)

            sam.train()

            if misc.is_main_process():
                with open(os.path.join(args.output, "log.txt"), "a") as f:
                    f.write("epoch: {}\n".format(epoch))
                    for k, v in train_stats.items():
                        f.write(f"{k}: {v}\n")
                    f.write("LRs:\n")
                    for k, v in current_lr_dict.items():
                        f.write(f"{k}: {v}\n")
                    f.write("\n")

        if epoch % args.model_save_fre == 0:
            model_name = "epoch_" + str(epoch) + ".pth"
            save_path = os.path.join(args.output, model_name)
            print("Saving model at", save_path)
            misc.save_on_master(sam.state_dict(), save_path)

    final_model_path = os.path.join(args.output, "final_model.pth")
    misc.save_on_master(sam.state_dict(), final_model_path)
    print("Training Reaches The Maximum Epoch Number")


@torch.no_grad()
def evaluate(args, sam, valid_dataloaders, epoch=0):
    sam.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    current_contrastive_weight = args.contrastive_weight if epoch < 50 else 0.0

    iou_thresholds = [i / 10.0 for i in range(1, 11)]

    final_miou_metric = misc.mIoU(nclass=1, threshold=0.5, from_logits=True)
    
    final_miou_metrics_by_threshold = {
        threshold: misc.mIoU(nclass=1, threshold=threshold, from_logits=True)
        for threshold in iou_thresholds
    }

    pd_fa_metric = misc.PD_FA(
        nclass=1,
        bins=10,
        match_distance=3,
        from_logits=True,
    )

    print("Validating...")
    for valid_dataloader in valid_dataloaders:
        metric_size = resolve_eval_size_from_loader(valid_dataloader, default_size=1024)

        for data_val in valid_dataloader:
            inputs_val = data_val["image"]
            labels_val = data_val["label"]
            captions_val = data_val["caption"]

            inputs_val = inputs_val.to(args.device, non_blocking=True)
            labels_val = labels_val.to(args.device, non_blocking=True)
            captions_val = move_caption_to_device(captions_val, args.device)

            if labels_val.dim() == 3:
                labels_val = labels_val.unsqueeze(1)
            labels_val = (labels_val > 0).float()
            labels_val_sup = resize_labels_for_supervision(labels_val, args)
            labels_val_metric = resize_labels_for_metric(labels_val, metric_size)
            labels_val_pd_fa = resize_labels_for_pd_fa(labels_val, target_size=256)

            batched_input_val = prepare_batched_input(
                inputs_val,
                labels_val,
                captions_val,
                scene_source=args.scene_source,
            )

            outputs = sam(
                batched_input_val,
                multimask_output=False,
            )
            pred_masks, text_tokens, image_tokens, aux_outputs = unpack_model_outputs(outputs)

            tta_preds = [pred_masks]
            for flip_dims in ([3], [2]):
                inputs_flip = torch.flip(inputs_val, dims=flip_dims)
                labels_flip = torch.flip(labels_val, dims=flip_dims)

                batched_input_flip = prepare_batched_input(
                    inputs_flip,
                    labels_flip,
                    captions_val,
                    scene_source=args.scene_source,
                )
                outputs_flip = sam(
                    batched_input_flip,
                    multimask_output=False,
                )
                pred_masks_flip, _, _, _ = unpack_model_outputs(outputs_flip)
                pred_masks_flip = torch.flip(pred_masks_flip, dims=flip_dims)
                tta_preds.append(pred_masks_flip)

            pred_masks_eval = torch.stack(tta_preds, dim=0).mean(dim=0)

            final_masks, final_loss = compute_mask_loss(
                pred_masks_eval,
                labels_val_sup,
                args,
            )

            final_masks_metric = resize_to_label_size(final_masks, labels_val_metric)

            final_masks_pd_fa = resize_to_label_size(final_masks, labels_val_pd_fa)

            _, _, val_contrast_loss = compute_alignment_losses(
                text_tokens=text_tokens,
                image_tokens=image_tokens,
                args=args,
                device=final_masks.device,
            )
            val_sam_aux_loss = compute_sam_aux_loss(
                aux_outputs=aux_outputs,
                labels=labels_val_sup,
                args=args,
                device=final_masks.device,
            )

            val_loss = (
                final_loss
                + args.sam_mask_weight * val_sam_aux_loss
                + current_contrastive_weight * val_contrast_loss
            )

            metric_update = dict(
                val_loss=val_loss.item(),
                val_final_loss=final_loss.item(),
                val_sam_aux_loss=val_sam_aux_loss.item(),
                val_contrast_loss=val_contrast_loss.item(),
            )
            metric_logger.update(**metric_update)

            final_miou_metric.update(final_masks_metric, labels_val_metric)
            for threshold_metric in final_miou_metrics_by_threshold.values():
                threshold_metric.update(final_masks_metric, labels_val_metric)
            pd_fa_metric.update(final_masks_pd_fa, labels_val_pd_fa)

    metric_logger.synchronize_between_processes()

    final_fa, final_pd = pd_fa_metric.get()

    pixAcc, miou = final_miou_metric.get()

    test_stats = {
        k: meter.global_avg
        for k, meter in metric_logger.meters.items()
        if meter.count > 0
    }

    test_stats["pixAcc@0.5"] = float(pixAcc)
    test_stats["mIoU@0.5"] = float(miou)

    iou_curve = {}
    for threshold in iou_thresholds:
        _, threshold_miou = final_miou_metrics_by_threshold[threshold].get()
        metric_name = f"mIoU@{threshold:.1f}"
        iou_curve[metric_name] = float(threshold_miou)
        test_stats[metric_name] = float(threshold_miou)

    print("pixAcc@0.5:", round(float(pixAcc), 6))
    print("mIoU@0.5:", round(float(miou), 6))
    print("IoU threshold sweep:", {k: round(v, 6) for k, v in iou_curve.items()})
    print("PD curve:", np.round(final_pd, 6))
    print("FA curve:", np.round(final_fa, 6))
    print("Averaged stats:", test_stats)

    return test_stats


if __name__ == "__main__":
    Train_NUAA_SIRST = {
        "name": "Train_NUAA_SIRST",
        "im_dir": "./data/NUAA-SIRST/train/images",
        "gt_dir": "./data/NUAA-SIRST/train/masks",
        "des_dir": "./data/NUAA-SIRST/train/descriptions",
        "im_ext": ".jpg",
        "gt_ext": "_pixels0.png",
        "des_ext": "_description.txt",
    }

    Test_NUAA_SIRST = {
        "name": "Test_NUAA_SIRST",
        "im_dir": "./data/NUAA-SIRST/test/images",
        "gt_dir": "./data/NUAA-SIRST/test/masks",
        "des_dir": "./data/NUAA-SIRST/test/descriptions",
        "im_ext": ".jpg",
        "gt_ext": "_pixels0.png",
        "des_ext": "_description.txt",
    }

    Train_NUDT_SIRST = {
        "name": "Train_NUDT_SIRST",
        "im_dir": "./data/NUDT-SIRST/train/images",
        "gt_dir": "./data/NUDT-SIRST/train/masks",
        "des_dir": "./data/NUDT-SIRST/train/descriptions",
        "im_ext": ".jpg",
        "gt_ext": ".png",
        "des_ext": "_description.txt",
    }

    Test_NUDT_SIRST = {
        "name": "Test_NUDT_SIRST",
        "im_dir": "./data/NUDT-SIRST/test/images",
        "gt_dir": "./data/NUDT-SIRST/test/masks",
        "des_dir": "./data/NUDT-SIRST/test/descriptions",
        "im_ext": ".jpg",
        "gt_ext": ".png",
        "des_ext": "_description.txt",
    }

    Train_XD_SIRST = {
        "name": "Train_XD_SIRST",
        "im_dir": "./data/XD-SIRST/train/images",
        "gt_dir": "./data/XD-SIRST/train/masks",
        "des_dir": "./data/XD-SIRST/train/descriptions",
        "im_ext": ".png",
        "gt_ext": ".png",
        "des_ext": "_description.txt",
    }

    Test_XD_SIRST = {
        "name": "Test_XD_SIRST",
        "im_dir": "./data/XD-SIRST/test/images",
        "gt_dir": "./data/XD-SIRST/test/masks",
        "des_dir": "./data/XD-SIRST/test/descriptions",
        "im_ext": ".png",
        "gt_ext": ".png",
        "des_ext": "_description.txt",
    }

    train_datasets = [Train_NUAA_SIRST]
    valid_datasets = [Test_NUAA_SIRST]

    args = get_args_parser()
    main(train_datasets, valid_datasets, args)
