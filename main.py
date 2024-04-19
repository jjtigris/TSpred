from encoder import get_preprocessing_fn
from dataset import Dataset
from model import DeepLabV3
from power_jaccard import JaccardLoss
from metric import IoU, Fscore
import torch

preprocessing_fn = get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
model = DeepLabV3(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, classes=no_classes, activation='softmax2d')

train_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    # augmentation=get_training_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
    resample_imbalanced=True
)

valid_dataset = Dataset(
    x_valid_dir, 
    y_valid_dir, 
    # augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
    resample_imbalanced=True
)

test_dataset = Dataset(
    x_test_dir,
    y_test_dir,
    classes=CLASSES,
    preprocessing=get_preprocessing(preprocessing_fn),
)
train_loader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=12)
valid_loader = DataLoader(valset, batch_size=8, shuffle=False, num_workers=4)
test_dataloader = DataLoader(test_dataset, shuffle=False)

loss = JaccardLoss()
metrics = [IoU(threshold=0.7), Fscore(threshold=0.7)]
optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=1e-5),
])

train_epoch = TrainEpoch(
    model=model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = ValidEpoch(
    model=model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)

test_epoch = ValidEpoch(
    model=best_model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
)
max_score = 0
val_losses = []
train_losses = []
val_iou = []
train_iou = []
val_f = []
train_f = []

for i in range(0, 20):
    
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)

    train_losses.append(train_logs['jaccard_loss'])
    train_iou.append(train_logs['iou_score'])
    train_f.append(train_logs['fscore'])

    val_losses.append(valid_logs['jaccard_loss'])
    val_iou.append(valid_logs['iou_score'])
    val_f.append(valid_logs['fscore'])
    
    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']

logs = test_epoch.run(test_dataloader)