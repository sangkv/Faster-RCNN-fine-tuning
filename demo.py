import os
import time

import torch
from torchvision.transforms import functional as F
from torchvision.transforms import v2 as Tv2
from PIL import Image, ImageDraw

import utils


# DEVICE
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# DATA
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

def draw_boxes_on_image(img, targets, category_colors=category_colors, score_threshold=0.5):
    '''Draw bounding boxes on the PIL image.'''
    draw = ImageDraw.Draw(img)

    if {'boxes', 'labels', 'scores'}.issubset(targets.keys()):
        boxes = targets['boxes'].cpu().numpy()
        labels = targets['labels'].cpu().numpy()
        scores = targets['scores'].cpu().numpy()

        for bbox, label, score in zip(boxes, labels, scores):
            if score >= score_threshold:  # Only draw boxes with a confidence score >= score_threshold
                x1, y1, x2, y2 = bbox
                color = category_colors.get(label, 'gray')  # Use gray for unmapped classes
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

def get_transform(train):
    transforms = []
    '''
    if train:
        transforms.append(Tv2.RandomHorizontalFlip(0.5))
    '''
    transforms.append(Tv2.ToDtype(torch.float, scale=True))
    transforms.append(Tv2.ToPureTensor())
    return Tv2.Compose(transforms)

fixed_size = (500, 320)
test_transform = get_transform(train=False)

path_test = 'data/test'
output_dir = 'data/test_output'
utils.mkdir(output_dir)
list_image = os.listdir(path_test)

# LOAD MODEL
model_dir = 'trained_model/script_model.pt'
model = torch.jit.load(model_dir).to(device).eval()

# PREDICT
sum_inference_time = 0
with torch.no_grad():
    for image_path in list_image:
        img_path = os.path.join(path_test, image_path)
        img = Image.open(img_path).convert('RGB')
        img = img.resize(fixed_size, Image.BILINEAR)
        
        t0 = time.time()
        image = F.to_tensor(img)
        image = test_transform(image)
        image = image.to(device)
        
        (_, output) = model([image])
        t1 = time.time()
        sum_inference_time += (t1-t0)
        
        targets = {k: v for k, v in output[0].items()}
        draw_boxes_on_image(img, targets)
        img.save(os.path.join(output_dir, os.path.basename(img_path)))
        
time_one_image = sum_inference_time/len(list_image)
print('Time cost for one image: ', time_one_image)
fps = float(1/time_one_image)
print("FPS = {} ".format(fps, '.1f') )
