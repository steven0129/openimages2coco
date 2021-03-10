import pandas as pd
import numpy as np
import csv

TRAIN_ANN_CSV = 'train-annotations-object-segmentation.csv'
VAL_ANN_CSV = 'validation-annotations-object-segmentation.csv'
CLASS_DESCRIPTION = 'oidv6-class-descriptions.csv'
CLASS_SEG = 'classes-segmentation.txt'
SUBSET_ID = 0
CONFIG_DIR = f'val_{SUBSET_ID}_config'

training_set = pd.read_csv(TRAIN_ANN_CSV)
val_set = pd.read_csv(VAL_ANN_CSV)

training_set_num = len(training_set.index)
val_set_num = len(val_set.index)

training_set_filter = np.array([False for _ in range(training_set_num)])
val_set_filter = np.array([False for _ in range(val_set_num)])

label_map = {}

with open(CLASS_DESCRIPTION, newline='') as FILE:
    rows = csv.reader(FILE)

    for row in rows:
        label_map[row[0]] = row[1]

with open(CLASS_SEG, newline='') as FILE:
    for row in FILE:
        row = row.strip()

valid_labels = [
    '/m/04_sv',  # Motorcycle
    '/m/0ph39',  # Canoe
    '/m/03k3r',  # Horse
    '/m/01g317', # Person
    '/m/0cmf2',  # Airplane
    '/m/0dbzx',  # Mule
    '/m/0k5j',   # Aircraft
    '/m/01xq0k1', # Cattle
    '/m/01btn',  # Barge
    '/m/01xs3r', # Jet ski
    '/m/04yx4',  # Man
    '/m/01x3jk',  # Snowmobile
    '/m/07cmd',  # Tank
    '/m/03bt1vf',  # Woman
    '/m/02p0tk3',  # Human Body
    '/m/0cnyhnx',  # Bull
    '/m/05r655',   # Girl
    '/m/01bl7v',   # Boy
    '/m/07r04',    # Truck
    '/m/015p6',    # Bird
    '/m/0pg52',    # Taxi
    '/m/0898b',    # Zebra
    '/m/0k4j',     # Car
    '/m/0gv1x',    # Parrot
    '/m/01f8m5',   # Blue jay
    '/m/01lcw4',   # Limousine
    '/m/0ccs93',   # Canary
    '/m/01dy8n',  # Woodpecker
    '/m/0bt9lr',  # Dog
    '/m/0h23m',   # Sparrow
    '/m/01bjv',   # Bus
    '/m/09csl',   # Eagle
    '/m/01yrx',   # Cat
    '/m/0f6wt',   # Falcon
    '/m/0h2r6',   # Van
]

for valid_label in valid_labels:
    curr_train_filter = np.logical_and( (training_set['LabelName'] == valid_label).to_numpy()  ,  training_set['ImageID'].str.startswith(str(SUBSET_ID)) )
    curr_val_filter =  np.logical_and(  (val_set['LabelName'] == valid_label).to_numpy()  ,  val_set['ImageID'].str.startswith(str(SUBSET_ID)) )

    open(f'{CONFIG_DIR}/train-statistics.txt', 'a+').write(f'{label_map[valid_label]},{np.sum(curr_train_filter)}\n')
    open(f'{CONFIG_DIR}/val-statistics.txt', 'a+').write(f'{label_map[valid_label]},{np.sum(curr_val_filter)}\n')

    training_set_filter = np.logical_or(training_set_filter, curr_train_filter)
    val_set_filter = np.logical_or(val_set_filter, curr_val_filter)

training_set = training_set[training_set_filter]
val_set = val_set[val_set_filter]

print('Training Set:', len(training_set.index))
print('Val Set:', (len(val_set.index)))

train_img_id = 'train/' + training_set['ImageID']
val_img_id = 'validation/' + val_set['ImageID']

img_id = pd.concat([train_img_id, val_img_id])
img_id.to_csv(f'{CONFIG_DIR}/img_list.csv', index=False, header=False)
