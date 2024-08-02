import os
import json
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageDraw
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import v2 as Tv2
from torchvision.transforms import functional as F
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import utils


# DEVICE
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# DATASETS
classes_to_idx = {
    0: 'background',
    1: 'bottom_left',
    2: 'bottom_right',
    3: 'top_left',
    4: 'top_right',
    5: 'unknown'
}

category_colors = {
    0: 'black',
    1: 'red',
    2: 'orange',
    3: 'blue',
    4: 'purple',
    5: 'yellow'
}

@dataclass
class DatasetConfig:
    root: str
    annotations_file: str
    train_img_size: tuple
    subset: str = 'train'  # Default to 'train'
    transforms: any = None

class CustomDataset(Dataset):
    def __init__(self, config: DatasetConfig):
        self.root = config.root
        self.subset = config.subset
        self.annotations_file = os.path.join(self.root, self.subset, config.annotations_file)
        self.train_img_size = config.train_img_size
        self.transforms = config.transforms
        
        self.imgs = []
        self.img_annotations = {}
        
        self._load_annotations()

    def __len__(self):
        return len(self.imgs)

    def _load_annotations(self):
        with open(self.annotations_file, 'r') as f:
            data = json.load(f)

        image_id_to_filename = {image['id']: image['file_name'] for image in data['images']}
        added_image_ids = set()
        
        for annotation in data['annotations']:
            image_id = annotation['image_id']
            bbox = annotation['bbox']
            category_id = annotation['category_id']

            file_name = image_id_to_filename.get(image_id, '')
            image_path = os.path.join(self.root, self.subset, file_name)

            if os.path.isfile(image_path) and image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                if image_id not in added_image_ids:
                    self.imgs.append((image_path, image_id))
                    added_image_ids.add(image_id)
                
                if image_id not in self.img_annotations:
                    self.img_annotations[image_id] = {'boxes': [], 'labels': []}

                self.img_annotations[image_id]['boxes'].append(bbox)
                self.img_annotations[image_id]['labels'].append(category_id)

    def __getitem__(self, idx):
        img_path, image_id = self.imgs[idx]
        if not os.path.exists(img_path):
            default_img = torch.zeros(3, *self.train_img_size)  # 3 color channels
            default_target = {'boxes': torch.tensor([[0, 0, 0, 0]], dtype=torch.float32),
                              'labels': torch.tensor([0], dtype=torch.int64)}
            return default_img, default_target

        img = Image.open(img_path).convert('RGB')
        orig_width, orig_height = img.size
        img = img.resize(self.train_img_size, Image.BILINEAR)
        img = F.to_tensor(img)

        scale_x = self.train_img_size[0] / orig_width
        scale_y = self.train_img_size[1] / orig_height

        annotations = self.img_annotations.get(image_id, {'boxes': [], 'labels': []})
        if annotations['boxes']:
            scaled_boxes = [
                [
                    max(0, min(bbox[0] * scale_x, self.train_img_size[0])),
                    max(0, min(bbox[1] * scale_y, self.train_img_size[1])),
                    max(0, min((bbox[0] + bbox[2]) * scale_x, self.train_img_size[0])),
                    max(0, min((bbox[1] + bbox[3]) * scale_y, self.train_img_size[1]))
                ]
                for bbox in annotations['boxes']
            ]
            labels = annotations['labels']
        else:
            scaled_boxes = [[0, 0, 0, 0]]
            labels = [0]

        boxes = torch.tensor(scaled_boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels}
        if self.transforms:
            img, target = self.transforms(img, target)
        return img, target

def get_transform(train):
    transforms = []
    '''
    if train:
        transforms.append(Tv2.RandomHorizontalFlip(0.5))
    '''
    transforms.append(Tv2.ToDtype(torch.float, scale=True))
    transforms.append(Tv2.ToPureTensor())
    return Tv2.Compose(transforms)

root = 'data'
annotations_json = 'result.json'
fixed_size = (500, 320)

# Configuration for training and validation datasets
train_config = DatasetConfig(root,
                             annotations_file=annotations_json,
                             train_img_size=fixed_size,
                             subset='train',
                             transforms=get_transform(train=True))
val_config = DatasetConfig(root,
                           annotations_file=annotations_json,
                           train_img_size=fixed_size,
                           subset='val',
                           transforms=get_transform(train=False))

train_dataset = CustomDataset(train_config)
val_dataset = CustomDataset(val_config)

print(f"Length of Train Dataset: {len(train_dataset)}")
print(f"Length of Validation Dataset: {len(val_dataset)}")

# DATA LOADERS
def collate_fn(batch):
    imgs, targets = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    real_targets = []
    for target in targets:
        # Filter out dummy boxes
        mask = target['boxes'].sum(dim=1) > 0
        real_targets.append({'boxes': target['boxes'][mask], 'labels': target['labels'][mask]})
    return imgs, real_targets

# Define training and validation data loaders
train_data_loader = DataLoader(
    train_dataset,
    batch_size=2, 
    shuffle=True, 
    collate_fn=collate_fn
)

val_data_loader = DataLoader(
    val_dataset, 
    batch_size=1, 
    shuffle=False, 
    collate_fn=collate_fn
)

# Check data loaders
batch = next(iter(train_data_loader))
print(len(batch))

# TRAIN
def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        
        with torch.autocast(device_type='cuda', enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

def visualize_predictions(images, outputs, targets, epoch, save_dir):
    utils.mkdir(save_dir)
    for idx, (image, output, target) in enumerate(zip(images, outputs, targets)):
        try:
            image = image.permute(1, 2, 0).cpu().numpy()  # Convert image tensor to numpy array
            image = np.clip(image * 255, 0, 255).astype(np.uint8)  # Clip image values and convert to uint8
            pil_image = Image.fromarray(image)
            draw = ImageDraw.Draw(pil_image)
            
            # Draw ground truth boxes
            for box, label in zip(target['boxes'], target['labels']):
                box = box.cpu().numpy().astype(int)
                xmin, ymin, xmax, ymax = box
                color = category_colors[label.item()]
                draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=2)
                draw.text((xmin, ymin), f'{classes_to_idx[label.item()]} ({label.item()})', fill=color)
            
            # Draw predicted boxes
            for box, label in zip(output['boxes'], output['labels']):
                box = box.cpu().numpy().astype(int)
                xmin, ymin, xmax, ymax = box
                color = 'red'
                draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=2)
                draw.text((xmin, ymin), f'Pred: {label.item()}', fill=color)

            save_path = os.path.join(save_dir, f'prediction_epoch_{epoch}_image_{idx}.png')
            pil_image.save(save_path)
        except Exception as e:
            print(f"Error while visualizing prediction for index {idx}: {e}")

@torch.inference_mode()
def evaluate(model, data_loader, device, epoch, save_dir, n_samples=10):
    model.eval()
    metric = MeanAveragePrecision(iou_type='bbox')
    samples = []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(images)

        # Convert outputs for torchmetrics
        preds = [
            {'boxes': out['boxes'], 'scores': out['scores'], 'labels': out['labels']}
            for out in outputs
        ]
        targs = [
            {'boxes': tgt['boxes'], 'labels': tgt['labels']}
            for tgt in targets
        ]

        # Update metric for mAP calculation
        metric.update(preds, targs)

        # Collect samples for visualization (limit to n_samples)
        if len(samples) < n_samples:
            for img, out, tgt in zip(images, outputs, targets):
                samples.append((img, out, tgt))
                if len(samples) >= n_samples:
                    break

    # Visualize predictions
    visualize_predictions([s[0] for s in samples], [s[1] for s in samples], [s[2] for s in samples], epoch, save_dir)

    # Compute and print results
    results = metric.compute()
    print("mAP results :", results)
    
    return results

def get_model(num_classes):
    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights='DEFAULT')

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

# replace the classifier with a new one, that has num_classes which is user-defined
num_classes=len(classes_to_idx)
# get the model using our helper function
model = get_model(num_classes)
# move model to the right device
model.to(device)
print(model)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.8
)

# Number of epochs
num_epochs = 100

# Initialize best mAP
best_map = -float('inf')

# Automatic Mixed Precision
scaler = torch.GradScaler('cuda')

utils.mkdir('trained_model')
for epoch in range(num_epochs):
    print("------------------------------------------")
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=10, scaler=None)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the validation dataset
    results = evaluate(model, val_data_loader, device, epoch, save_dir='data/val_output')
    
    # Save the model checkpoint if it's the best mAP
    current_map = results['map'].item()
    if current_map > best_map:
        best_map = current_map
        torch.save(model.state_dict(), 'trained_model/best_model.pth')

print("That's it!")
print(best_map)

# SAVE MODEL
checkpoint_path = 'trained_model/best_model.pth'
# Function to load the trained model
def load_model(num_classes, checkpoint_path, device):
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, num_classes=num_classes, box_nms_thresh=0.0)
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    return model

test_model = load_model(len(classes_to_idx), checkpoint_path, torch.device('cpu'))
# Convert model for deploy
traced_model = torch.jit.script(test_model)
traced_model.save(f'trained_model/script_model.pt')
print('Model saved!')

