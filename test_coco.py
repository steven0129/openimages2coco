import json
from pycocotools.coco import COCO
import json
from tqdm import tqdm

annFile = f'custom_train_open01.json'
jsonObj = json.load(open(annFile, 'r'))
coco = COCO(annFile)
cats = coco.loadCats(coco.getCatIds())
names = [cat['name'] for cat in cats]
print(f'COCO categories: {", ".join(names)}')

names = set([cat['supercategory'] for cat in cats])
print(f'COCO supercategories: {", ".join(names)}')

my_classes = ['person', 'bicycle', 'car', 'motorbike', 'airplane', 'ship', 'bird', 'cat', 'dog', 'horse', 'cow']
my_classes = my_classes[:2]
new_coco = {
    'info': {},
    'licenses': [],
    'images': [],
    'annotations': [],
    'categories': []
}

for cls in tqdm(my_classes):
    catIds = coco.getCatIds(catNms=[cls])
    imgIds = coco.getImgIds(catIds=catIds)
    imgs = coco.loadImgs(imgIds)
    annIds = coco.getAnnIds(imgIds=imgIds, catIds=catIds, iscrowd=False)
    anns = coco.loadAnns(ids=annIds)
    print(len(anns))
