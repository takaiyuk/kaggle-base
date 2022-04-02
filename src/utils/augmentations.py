import copy
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch


@dataclass
class MixUpLabelData:
    labels_a: torch.tensor
    labels_b: torch.tensor
    lam: np.array


@dataclass
class CutMixLabelData:
    labels_a: torch.tensor
    labels_b: torch.tensor
    lam: np.array


class BaseAugmentation(metaclass=ABCMeta):
    @abstractmethod
    def augment(self):
        pass

    @abstractmethod
    def loss(self):
        pass


class MixUp(BaseAugmentation):
    def augment(
        batch: Dict[str, np.array],
        alpha: float = 1.0,
        seed: float = 42,
    ) -> Tuple[np.array, MixUpLabelData]:
        images, labels = batch["image"], batch["label"]

        ### Shuffle Minibatch ###
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        indices = torch.randperm(images.size(0))
        images_s, labels_s = images[indices], labels[indices]

        lam = np.random.beta(alpha, alpha)
        images_mixup = lam * images + (1 - lam) * images_s
        labels_mixup = MixUpLabelData(labels_a=labels, labels_b=labels_s, lam=lam)

        return images_mixup, labels_mixup

    def loss(
        self,
        criterion: Any,
        preds: torch.Tensor,
        labels: MixUpLabelData,
        device: str = "cuda",
    ) -> float:
        labels_a, labels_b, lam = (
            labels.labels_a.to(device),
            labels.labels_b.to(device),
            labels.lam,
        )
        loss = lam * criterion(preds, labels_a) + (1 - lam) * criterion(preds, labels_b)
        return loss


class CutOut(BaseAugmentation):
    def augment(
        self,
        batch: Dict[str, np.array],
        seed: float = 42,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        images, labels = batch["image"], batch["label"]

        images_s = torch.zeros_like(images)
        np.random.seed(seed)
        lam = np.random.uniform(0.0, 1.0)

        H, W = images.shape[2:]
        r_x = np.random.uniform(0, W)
        r_y = np.random.uniform(0, H)
        r_w = W * np.sqrt(1 - lam)
        r_h = H * np.sqrt(1 - lam)
        x1 = int(np.round(max(r_x - r_w / 2, 0)))
        x2 = int(np.round(min(r_x + r_w / 2, W)))
        y1 = int(np.round(max(r_y - r_h / 2, 0)))
        y2 = int(np.round(min(r_y + r_h / 2, H)))

        images_cutout = copy.deepcopy(images)
        images_cutout[:, :, x1:x2, y1:y2] = images_s[:, :, x1:x2, y1:y2]

        return images_cutout, labels

    def loss(self, criterion: Any, preds: torch.Tensor, labels: torch.Tensor) -> float:
        loss = criterion(preds, labels)
        return loss


class CutMix(BaseAugmentation):
    def augment(
        self,
        batch: Dict[str, np.array],
        seed: float = 42,
    ) -> Tuple[np.array, CutMixLabelData]:
        images, labels = batch["image"], batch["label"]

        ### Shuffle Minibatch ###
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        indices = torch.randperm(images.size(0))
        images_s, labels_s = images[indices], labels[indices]

        np.random.seed(seed)
        lam = np.random.uniform(0.0, 1.0)

        H, W = images.shape[2:]
        r_x = np.random.uniform(0, W)
        r_y = np.random.uniform(0, H)
        r_w = W * np.sqrt(1 - lam)
        r_h = H * np.sqrt(1 - lam)
        x1 = int(np.round(max(r_x - r_w / 2, 0)))
        x2 = int(np.round(min(r_x + r_w / 2, W)))
        y1 = int(np.round(max(r_y - r_h / 2, 0)))
        y2 = int(np.round(min(r_y + r_h / 2, H)))

        images_cutmix = copy.deepcopy(images)
        images_cutmix[:, :, x1:x2, y1:y2] = images_s[:, :, x1:x2, y1:y2]
        labels_cutmix = CutMixLabelData(labels_a=labels, labels_b=labels_s, lam=lam)

        return images_cutmix, labels_cutmix

    def loss(
        self,
        criterion: Any,
        preds: torch.Tensor,
        labels: MixUpLabelData,
        device: str = "cuda",
    ) -> float:
        labels_a, labels_b, lam = (
            labels.labels_a.to(device),
            labels.labels_b.to(device),
            labels.lam,
        )
        loss = lam * criterion(preds, labels_a) + (1 - lam) * criterion(preds, labels_b)
        return loss
