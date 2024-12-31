import os
import logging
import warnings
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR) 
import paddle
import paddle.vision.transforms as T
from paddle.io import Dataset, DataLoader
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from paddle.vision.models import resnet50
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from tqdm import tqdm
import paddle.io as io
paddle.disable_static() 
os.environ['GLOG_v'] = '0'  
os.environ['GLOG_logtostderr'] = '0'  

class SubsetDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

class SmokeDataset(Dataset):
    def __init__(self, img_dir, anno_dir, transform=None):
        self.img_dir = img_dir
        self.anno_dir = anno_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        anno_name = img_name.replace('.jpg', '.xml')
        anno_path = os.path.join(self.anno_dir, anno_name)
        
        tree = ET.parse(anno_path)
        root = tree.getroot()
        
        obj = root.find('object')
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        img_width = float(root.find('size').find('width').text)
        img_height = float(root.find('size').find('height').text)
        
        bbox = [xmin/img_width, ymin/img_height, xmax/img_width, ymax/img_height]
        
        if self.transform:
            img = self.transform(img)
        
        return img, paddle.to_tensor(bbox)

class SmokeDetector(paddle.nn.Layer):
    def __init__(self):
        super(SmokeDetector, self).__init__()
        
        backbone = resnet50(pretrained=True)
        self.features = paddle.nn.Sequential(*list(backbone.children())[:-2])
        
        self.detector = paddle.nn.Sequential(
            paddle.nn.Conv2D(2048, 512, 1),
            paddle.nn.ReLU(),
            paddle.nn.AdaptiveAvgPool2D(1),
            paddle.nn.Flatten(),
            paddle.nn.Linear(512, 4)
        )
        
    def forward(self, x):
        features = self.features(x)
        bbox = self.detector(features)
        return bbox

def calculate_iou(pred_box, target_box):
    xmin_pred, ymin_pred, xmax_pred, ymax_pred = pred_box
    xmin_gt, ymin_gt, xmax_gt, ymax_gt = target_box
    
    inter_xmin = max(xmin_pred, xmin_gt)
    inter_ymin = max(ymin_pred, ymin_gt)
    inter_xmax = min(xmax_pred, xmax_gt)
    inter_ymax = min(ymax_pred, ymax_gt)
    
    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    
    pred_area = (xmax_pred - xmin_pred) * (ymax_pred - ymin_pred)
    gt_area = (xmax_gt - xmin_gt) * (ymax_gt - ymin_gt)
    
    union_area = pred_area + gt_area - inter_area
    
    return inter_area / union_area

def evaluate_model(model, val_loader):
    model.eval()
    all_ious = []
    
    with paddle.no_grad():
        for images, targets in val_loader:
            outputs = model(images)
            
            for pred, target in zip(outputs.numpy(), targets.numpy()):
                iou = calculate_iou(pred, target)
                all_ious.append(iou)
    
    # Calculate metrics
    iou_threshold = 0.5
    true_positives = sum(1 for iou in all_ious if iou >= iou_threshold)
    total_predictions = len(all_ious)
    
    precision = true_positives / total_predictions
    recall = true_positives / total_predictions  # In this case, each image has one box
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    return precision, recall, f1_score, np.mean(all_ious)

def plot_metrics(metrics_history):
    epochs = range(1, len(metrics_history['precision']) + 1)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(epochs, metrics_history['precision'], 'b-', label='Precision')
    plt.title('Precision over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(epochs, metrics_history['recall'], 'r-', label='Recall')
    plt.title('Recall over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(epochs, metrics_history['f1'], 'g-', label='F1 Score')
    plt.title('F1 Score over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(epochs, metrics_history['mIoU'], 'y-', label='mIoU')
    plt.title('Mean IoU over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

def train():
    img_dir = "pp_smoke/images"
    anno_dir = "pp_smoke/Annotations"
    
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = SmokeDataset(img_dir, anno_dir, transform)
    
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True)
    
    metrics_history = {
        'precision': [],
        'recall': [],
        'f1': [],
        'mIoU': [],
        'loss': []
    }
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(range(len(full_dataset)))):
        print(f'\nFOLD {fold+1}/{k_folds}')
        print('-' * 50)
        
        train_dataset = SubsetDataset(full_dataset, train_ids)
        val_dataset = SubsetDataset(full_dataset, val_ids)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        
        model = SmokeDetector()
        criterion = paddle.nn.MSELoss()
        optimizer = paddle.optimizer.Adam(parameters=model.parameters())
        
        num_epochs = 50
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            
            # Training progress bar
            train_bar = tqdm(train_loader, 
                           desc=f'Epoch {epoch+1}/{num_epochs}',
                           ncols=120)
            
            for batch_idx, (images, targets) in enumerate(train_bar):
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()
                
                total_loss += loss.item()
                current_loss = total_loss / (batch_idx + 1)
                
                # Update training metrics in progress bar
                train_bar.set_postfix({
                    'Loss': f'{current_loss:.4f}'
                })
            
            # Evaluation phase
            model.eval()
            all_ious = []
            val_loss = 0
            
            val_bar = tqdm(val_loader, 
                         desc='Validating',
                         ncols=120,
                         leave=False)
            
            with paddle.no_grad():
                for batch_idx, (images, targets) in enumerate(val_bar):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    
                    for pred, target in zip(outputs.numpy(), targets.numpy()):
                        iou = calculate_iou(pred, target)
                        all_ious.append(iou)
                    
                    current_val_loss = val_loss / (batch_idx + 1)
                    current_miou = np.mean(all_ious)
                    
                    # Update validation metrics in progress bar
                    val_bar.set_postfix({
                        'Val_Loss': f'{current_val_loss:.4f}',
                        'mIoU': f'{current_miou:.4f}'
                    })
            
            # Calculate final metrics
            avg_loss = total_loss / len(train_loader)
            val_loss = val_loss / len(val_loader)
            
            iou_threshold = 0.5
            true_positives = sum(1 for iou in all_ious if iou >= iou_threshold)
            total_predictions = len(all_ious)
            
            precision = true_positives / total_predictions
            recall = true_positives / total_predictions
            f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
            miou = np.mean(all_ious)
            
            # Store metrics
            metrics_history['precision'].append(precision)
            metrics_history['recall'].append(recall)
            metrics_history['f1'].append(f1_score)
            metrics_history['mIoU'].append(miou)
            metrics_history['loss'].append(avg_loss)
            
            # Print epoch summary
            print(f'\nEpoch {epoch+1}/{num_epochs} Summary:')
            print(f'Training Loss: {avg_loss:.4f}')
            print(f'Validation Loss: {val_loss:.4f}')
            print(f'Precision: {precision:.4f}')
            print(f'Recall: {recall:.4f}')
            print(f'F1 Score: {f1_score:.4f}')
            print(f'mIoU: {miou:.4f}')
            print('-' * 50)
            
            if (epoch + 1) % 10 == 0:
                paddle.save(model.state_dict(), 
                          f'smoke_detector_fold{fold+1}_epoch_{epoch+1}.pdparams')
        
        paddle.save(model.state_dict(), 
                   f'smoke_detector_fold{fold+1}_final.pdparams')
    
    # Plot and save metrics
    plot_metrics(metrics_history)
    np.save('metrics_history.npy', metrics_history)

if __name__ == "__main__":
    train()