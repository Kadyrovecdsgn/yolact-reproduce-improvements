import torch
import numpy as np
import random

def mixup_collate_fn(batch, alpha=1.0):
    """
    Функция collate для DataLoader, которая применяет MixUp к батчу.
    
    Аргументы:
        batch: список элементов (img, (boxes, masks, labels, ...))
               где img - тензор [C, H, W]
               boxes: [num_gt, 4], masks: [num_gt, H, W], labels: [num_gt]
        alpha: параметр бета-распределения (1.0 = равномерное)
    
    Возвращает:
        images: [batch_size, C, H, W] смешанные изображения
        targets: список кортежей (boxes, masks, labels) с объединенными GT
    """
    images = []
    targets = []
    
    # Извлекаем изображения и цели из батча
    for img, target in batch:
        images.append(img)
        targets.append(target)
    
    images = torch.stack(images, dim=0)
    
    # Генерируем коэффициенты смешивания для всех изображений в батче
    # Для каждого изображения в батче создаем пару со случайным другим
    batch_size = images.size(0)
    lam = np.random.beta(alpha, alpha, size=batch_size)
    indices = torch.randperm(batch_size)
    
    mixed_images = []
    mixed_targets = []
    
    for i in range(batch_size):
        lam_i = lam[i]
        j = indices[i]
        
        # Смешиваем i-ое и j-ое изображения
        mixed_img = lam_i * images[i] + (1 - lam_i) * images[j]
        mixed_images.append(mixed_img)
        
        # Комбинируем цели (boxes, masks, labels)
        boxes_i, masks_i, labels_i = targets[i][:3]  # YOLACT хранит (boxes, masks, labels)
        boxes_j, masks_j, labels_j = targets[j][:3]
        
        # Объединяем GT объекты (просто конкатенируем)
        # Важно: маски могут иметь разный размер (H,W) — они у всех одинаковые, т.к. исходные изображения приведены к единому размеру
        mixed_boxes = torch.cat([boxes_i, boxes_j], dim=0)
        mixed_masks = torch.cat([masks_i, masks_j], dim=0)
        mixed_labels = torch.cat([labels_i, labels_j], dim=0)
        

        
        mixed_targets.append((mixed_boxes, mixed_masks, mixed_labels))
    
    mixed_images = torch.stack(mixed_images, dim=0)
    
    return mixed_images, mixed_targets


class MixUpDatasetWrapper:
    """
    Обертка для датасета, которая модифицирует getitem для применения MixUp
    на уровне отдельных пар изображений (а не целого батча).
    Используется, если вы не хотите модифицировать DataLoader.
    """
    def __init__(self, dataset, alpha=1.0, prob=0.5):
        self.dataset = dataset
        self.alpha = alpha
        self.prob = prob
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        
        # С вероятностью prob применяем MixUp
        if random.random() < self.prob:
            # Выбираем другой случайный индекс
            j = random.randint(0, len(self.dataset) - 1)
            if j == idx:
                j = (j + 1) % len(self.dataset)
            img2, target2 = self.dataset[j]
            
            lam = np.random.beta(self.alpha, self.alpha)
            
            # Смешиваем изображения
            img = lam * img + (1 - lam) * img2
            
            # Объединяем GT
            boxes1, masks1, labels1 = target
            boxes2, masks2, labels2 = target2
            
            mixed_boxes = torch.cat([boxes1, boxes2], dim=0)
            mixed_masks = torch.cat([masks1, masks2], dim=0)
            mixed_labels = torch.cat([labels1, labels2], dim=0)
            
            target = (mixed_boxes, mixed_masks, mixed_labels)
        
        return img, target
    

def mixup_batch(batch, alpha=1.0, prob=0.5):
    # С вероятностью prob применяем MixUp
    if random.random() > prob:
        # Возвращаем оригинальные данные без изменений
        images = []
        targets = []
        masks = []
        num_crowds = []
        for img, (tgt, msk, nc) in batch:
            images.append(img)
            # Конвертация в тензор, если нужно
            if not isinstance(tgt, torch.Tensor):
                tgt = torch.from_numpy(tgt).float()
            if not isinstance(msk, torch.Tensor):
                msk = torch.from_numpy(msk).float()
            targets.append(tgt)
            masks.append(msk)
            num_crowds.append(nc)
        return images, targets, masks, num_crowds

    # Применяем MixUp
    images = []
    targets = []
    masks = []
    num_crowds = []

    for img, (tgt, msk, nc) in batch:
        images.append(img)
        if not isinstance(tgt, torch.Tensor):
            tgt = torch.from_numpy(tgt).float()
        if not isinstance(msk, torch.Tensor):
            msk = torch.from_numpy(msk).float()
        targets.append(tgt)
        masks.append(msk)
        num_crowds.append(nc)

    batch_size = len(images)
    lam = np.random.beta(alpha, alpha, size=batch_size)
    # indices = torch.randperm(batch_size)
    indices = torch.tensor(np.random.permutation(batch_size), device='cpu')

    mixed_images = []
    mixed_targets = []
    mixed_masks = []
    mixed_num_crowds = []

    for i in range(batch_size):
        j = indices[i]
        lam_i = lam[i]

        # Смешиваем изображения
        mixed_img = lam_i * images[i] + (1 - lam_i) * images[j]
        mixed_images.append(mixed_img)

        # Объединяем цели (боксы, метки) и маски
        mixed_tgt = torch.cat([targets[i], targets[j]], dim=0)
        mixed_msk = torch.cat([masks[i], masks[j]], dim=0)
        mixed_targets.append(mixed_tgt)
        mixed_masks.append(mixed_msk)
        mixed_num_crowds.append(num_crowds[i] + num_crowds[j])

    return mixed_images, mixed_targets, mixed_masks, mixed_num_crowds