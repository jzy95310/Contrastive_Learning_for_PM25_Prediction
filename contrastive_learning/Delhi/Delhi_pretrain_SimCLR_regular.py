import sys
import torch
sys.path.insert(0, '../../model_utils')
sys.path.insert(0, '../../contrastive_models/simclr_custom/simclr')

from torch.utils.data import DataLoader
from contrastive_utils import MyCLImageDataset
for _ in range(2):
    try:
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks.progress import TQDMProgressBar
        from pytorch_lightning.callbacks import EarlyStopping
        from simclr_module_resnet50 import SimCLR
        from simclr_transforms import SimCLREvalDataTransform, SimCLRTrainDataTransform
    except Exception as err:
        pass

def cli_main():
    input_height = 100
    train_dataset_SSL = MyCLImageDataset(root_dir='../../data/Delhi_labeled.pkl', mode='train',
                                      transform=SimCLRTrainDataTransform(input_height=input_height))
    val_dataset_SSL = MyCLImageDataset(root_dir='../../data/Delhi_labeled.pkl', mode='val', 
                                    transform=SimCLREvalDataTransform(input_height=input_height), train_val_ratio=0.75)
    train_size = len(train_dataset_SSL)
    
    lr = 1e-4
    train_dataloader_SSL = DataLoader(train_dataset_SSL, batch_size=32, num_workers=11, shuffle=True)
    val_dataloader_SSL = DataLoader(val_dataset_SSL, batch_size=32, num_workers=11, shuffle=False)
    
    # SimCLR Model
    simclr_model = SimCLR(gpus=1, batch_size=32, num_samples=train_size, lr=lr, optimizer='lars', arch='resnet50', hidden_mlp=2048)
    
    # Training
    max_epochs = 250   # If not specified, the default training_epochs is 1000
    bar = TQDMProgressBar(refresh_rate=500)
#     checkpoint_callback = ModelCheckpoint(
#         filepath='./model_checkpoint/Intermediate_checkpoints/{epoch:02d}-{val_loss:.2f}',
#         period=50
#     )
    # early_stop_callback = EarlyStopping(monitor='avg_val_loss')
    trainer = pl.Trainer(
        gpus=1, 
        max_epochs=max_epochs, 
        callbacks=[bar], 
        # checkpoint_callback=checkpoint_callback, 
    )
    trainer.fit(model=simclr_model, train_dataloaders=train_dataloader_SSL, val_dataloaders=val_dataloader_SSL)
    
    # Save encoder parameters
    simclr_encoder = simclr_model.encoder
    torch.save(simclr_encoder.state_dict(), '../../model_checkpoint/encoder_params_resnet50_regular_Delhi_SimCLR.pkl')

if __name__ == '__main__':
    cli_main()