import pandas as pd
import cv2
import numpy as np
import datetime
from PIL import Image
from tqdm import tqdm

train_ann = pd.read_csv('train-annotations-object-segmentation.csv')
val_ann = pd.read_csv('validation-annotations-object-segmentation.csv')
machine_label = pd.read_csv('train-annotations-machine-imagelabels.csv')

LABEL_OPEN2COCO = {
    '/m/04_sv': 4,  # Motorcycle
    '/m/0ph39': 9,  # Canoe (獨木舟)
    '/m/03k3r': 19,  # Horse
    '/m/01g317': 1, # Person
    '/m/0cmf2': 5,  # Airplane
    '/m/0dbzx': 19,  # Mule(馬騾)
    '/m/0k5j': 5,   # Aircraft
    '/m/01xq0k1': 21, # Cattle
    '/m/01btn': 9,  # Barge
    '/m/01xs3r': 9, # Jet ski
    '/m/04yx4': 1,  # Man
    '/m/01x3jk': 3,  # Snowmobile
    '/m/07cmd': 3,  # Tank
    '/m/03bt1vf': 1,  # Woman
    '/m/02p0tk3': 1,  # Human Body
    '/m/0cnyhnx': 21,  # Bull
    '/m/05r655': 1,   # Girl
    '/m/01bl7v': 1,   # Boy
    '/m/07r04': 3,    # Truck
    '/m/015p6': 16,    # Bird
    '/m/0pg52': 3,    # Taxi
    '/m/0898b': 19,    # Zebra
    '/m/0k4j': 3,     # Car
    '/m/0gv1x': 16,    # Parrot(鸚鵡)
    '/m/01f8m5': 16,   # Blue jay
    '/m/01lcw4': 3,   # Limousine(豪華轎車)
    '/m/0ccs93': 16,   # Canary(金絲雀)
    '/m/01dy8n': 16,  # Woodpecker(啄木鳥)
    '/m/0bt9lr': 18,  # Dog
    '/m/0h23m': 16,   # Sparrow
    '/m/01bjv': 3,   # Bus
    '/m/09csl': 16,   # Eagle
    '/m/01yrx': 17,   # Cat
    '/m/0f6wt': 16,  # Falcon(鶻)
    '/m/0h2r6': 3,   # Van
}

IMGID_OPEN2COCO = {}

def cnt2poly(contours):  # Contour --> Polygon
    segmentations = []
    for contour in contours:
        single_segmentation = []

        for single_point in contour:
            cnt_list = single_point.flatten().tolist()
            cnt_str = f'{float(cnt_list[0])},{float(cnt_list[1])}'
            single_segmentation.append(cnt_str)

        if len(single_segmentation) >= 4:
            segmentations.append('['  + ','.join(single_segmentation) + ']')

    if len(segmentations) == 0:
        return None
    else:
        return '[' + ','.join(segmentations) + ']'

def cnt2bbox(contours):  # Contour --> BBox (xywh)
    all_x = []
    all_y = []
    
    for single_contour in contours:
        single_contour = list(map(lambda x: x.flatten().tolist(), single_contour))
        all_x.extend(list(map(lambda x: x[0], single_contour)))
        all_y.extend(list(map(lambda x: x[1], single_contour)))

    xmin = min(all_x)
    ymin = min(all_y)
    xmax = max(all_x)
    ymax = max(all_y)

    w = xmax - xmin
    h = ymax - ymin

    if w < 10 and h < 10:
        return None
    else:
        return f'[{xmin},{ymin},{w},{h}]'

if __name__ == '__main__':
    info_str = '"info": {},'
    license_str = '"licenses": [],'
    ann_str = '"annotations": ['
    img_str = '"images": ['
    cat_str = """
    "categories": [
        {
            "supercategory": "person",
            "id": 1,
            "name": "person"
        },
        {
            "supercategory": "vehicle",
            "id": 2,
            "name": "bicycle"
        },
        {
            "supercategory": "vehicle",
            "id": 3,
            "name": "car"
        },
        {
            "supercategory": "vehicle",
            "id": 4,
            "name": "motorcycle"
        },
        {
            "supercategory": "vehicle",
            "id": 5,
            "name": "airplane"
        },
        {
            "supercategory": "vehicle",
            "id": 9,
            "name": "boat"
        },
        {
            "supercategory": "animal",
            "id": 16,
            "name": "bird"
        },
        {
            "supercategory": "animal",
            "id": 17,
            "name": "cat"
        },
        {
            "supercategory": "animal",
            "id": 18,
            "name": "dog"
        },
        {
            "supercategory": "animal",
            "id": 19,
            "name": "horse"
        },
        {
            "supercategory": "animal",
            "id": 21,
            "name": "cow"
        }
    ]
    """

    img_list = []
    counter = 0

    with open('set-config-0/img_list.csv') as FILE:
        for line in FILE:
            img_list.append(line)

    machine_label = machine_label[machine_label['ImageID'].isin(list(map(lambda x: x.split('/')[-1].strip(), img_list)))]

    for img_list_counter, line in enumerate(tqdm(img_list)):
        filename = line.strip().split('/')[1]
        subset_type = line.strip().split('/')[0]
        subset_id = filename[0]

        if subset_type == 'train':
            img_name = f'{subset_type}-{subset_id}/{filename}.jpg'
            ann = train_ann[train_ann['ImageID'] == filename]
            masks_path = list(map(lambda x: f'train-masks-{subset_id}/{x}', ann['MaskPath'].values))
        elif subset_type == 'validation':
            img_name = f'{subset_type}/{filename}.jpg'
            ann = val_ann[val_ann['ImageID'] == filename]
            masks_path = list(map(lambda x: f'val-masks-{subset_id}/{x}', ann['MaskPath'].values))
            
        IMGID_OPEN2COCO[filename] = str(img_list_counter).zfill(12)
        img_id = IMGID_OPEN2COCO[filename]
        img = Image.open(img_name)
        w, h = img.size
        img = np.array(img)
        labels = ann['LabelName'].values
        
        # build annoatations
        flag = 0
        for label, path in zip(labels, masks_path):
            curr_machine_label = machine_label
            curr_machine_label = curr_machine_label[curr_machine_label['ImageID'] == filename]
            curr_machine_label = curr_machine_label[curr_machine_label['LabelName'] == label]
            curr_machine_label = curr_machine_label[curr_machine_label['Confidence'] > 0.9]

            if (len(curr_machine_label.index) == 0):
                continue
            
            try:
                mask = Image.open(path)
            except Exception:
                continue
            
            mask = mask.resize((w, h))
            mask = np.array(mask, np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            areas = list(map(cv2.contourArea, contours))
            area_str = str(sum(areas))
            contours = list(map(lambda x: x[:, 0, :], contours))
            if len(contours) == 0:
                continue
            
            seg_str = cnt2poly(contours)
            bbox_str = cnt2bbox(contours)
            
            if label in LABEL_OPEN2COCO and seg_str != None and bbox_str != None:
                flag = 1
                cat_id = LABEL_OPEN2COCO[label]

                ann_single = f"""
                {{
                    "segmentation": {seg_str},
                    "area": {area_str},
                    "iscrowd": 0,
                    "image_id": {int(img_id)},
                    "bbox": {bbox_str},
                    "category_id": {cat_id},
                    "id": {counter}
                }}
                """

                ann_single += ','
                counter += 1
                ann_str += ann_single

        if flag == 1:
            # build images
            img_single = f"""
                {{
                    "license": 1,
                    "file_name": "{filename}.jpg",
                    "coco_url": "None",
                    "height": {img.shape[0]},
                    "width": {img.shape[1]},
                    "flickr_url": "None",
                    "data_captured": "{datetime.datetime.now()}",
                    "id": {int(img_id)}
                }}
            """

            img_single += ','
            img_str += img_single

    ann_str = ann_str[:-1]
    img_str = img_str[:-1]
    ann_str += '],'
    img_str += '],'

    open(f'lg_train{datetime.datetime.today().strftime("%Y%m%d")}_openimages-0.json', 'w').write(f"""
        {{
            {info_str}
            {license_str}
            {img_str}
            {ann_str}
            {cat_str}
        }}
    """)