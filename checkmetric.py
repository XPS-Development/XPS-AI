import torch
from torch.utils.data import DataLoader
from model.models.model_deeper import XPSModel
from model.train.dataset import XPSDataset
from model.train.metrics import IoU, Accuracy, Precision, Recall
from run_train_rv import load_params

seed, path_to_data, path_to_real_data, json_dir, train_params, synth_params = load_params()

model = XPSModel()
model.load_state_dict(torch.load('C:/Users/User/XPSAI/train_log_20251015_1257/model.pt'))
model.eval()
val_dataset = XPSDataset(path_to_real_data)
val_loader = DataLoader(val_dataset, batch_size=train_params['batch_size'], shuffle=False)

print(f"всего спектров: {len(val_dataset)}")


metrics = {
    'iou_peak': IoU(),
    'acc_peak': Accuracy(),
    'prec_peak': Precision(), 
    'rec_peak': Recall(),
    'iou_max': IoU(),
    'acc_max': Accuracy(),
    'prec_max': Precision(),
    'rec_max': Recall()
}

@torch.no_grad()
def calculate_metrics(model, dataloader, metrics):
    results = {name: 0.0 for name in metrics.keys()}
    
    for data, target in dataloader:
        output = model(data)
        
        for name, metric_fn in metrics.items():
            if 'peak' in name:
                results[name] += metric_fn(output[:, 0, :], target[:, 0, :]).item()
            else:  
                results[name] += metric_fn(output[:, 1, :], target[:, 1, :]).item()
    
    for name in results:
        results[name] /= len(dataloader)
    
    return results

pytorch_metrics = calculate_metrics(model, val_loader, metrics)

for metric, value in pytorch_metrics.items():
    print(f"{metric}: {value:.4f}")
    