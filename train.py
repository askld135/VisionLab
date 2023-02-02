from dataset import Raw2RgbDataset, denormalize
from torch.utils.data import DataLoader
from time import time
from torch.nn import MSELoss
from model import ResUNet
from torch.optim import Adam, lr_scheduler
import numpy as np
import torch
import os, sys
root = "RGBW_training_dataset_fullres/"

batch_size = 1
print(os.path.join(root, "input/train_RGBW_full_input_0dB"))
train_0dB_dataset = Raw2RgbDataset(raw_dir=os.path.join(root, "input/crops/image_0dB"),
                                   gt_dir=os.path.join(root, "input/crops/gt"),
                                   db_type='train'
                                   )

train_0dB_dataloader = DataLoader(dataset=train_0dB_dataset,
                                  batch_size=batch_size,
                                  shuffle=True
                                  )

train_24dB_dataset = Raw2RgbDataset(raw_dir=os.path.join(root, "input/crops/image_24dB"),
                                   gt_dir=os.path.join(root, "input/crops/gt"),
                                   db_type='train'
                                   )

train_24dB_dataloader = DataLoader(dataset=train_24dB_dataset,
                                  batch_size=batch_size,
                                  shuffle=True
                                  )

train_42dB_dataset = Raw2RgbDataset(raw_dir=os.path.join(root, "input/crops/image_42dB"),
                                   gt_dir=os.path.join(root, "input/crops/gt"),
                                   db_type='train'
                                   )

train_42dB_dataloader = DataLoader(dataset=train_42dB_dataset,
                                  batch_size=batch_size,
                                  shuffle=True
                                  )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#model
model = ResUNet(in_channels=1).to(device)
model.train()
model.cuda()

#configs
num_epoch = 100

#optimizer
optimizer = Adam(model.parameters(), lr=1e-4)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_epoch)
torch.optim.lr_scheduler.CosineAnnealingLR

#Loss
criterion = MSELoss()

num_train = len(train_0dB_dataset) // batch_size

start_t = time()
loss_list = []
for epoch in range(num_epoch):
    start = time()
    
    # for each train batch
    for batch_idx, (inputs, labels) in enumerate(train_0dB_dataloader):
        
        inputs = inputs.to(device)
        labels = labels.to(device) 
        
        preds = model(inputs) #net.forward(input)
        
        loss = criterion(preds, labels)
        
        optimizer.zero_grad()    # G = 0
        
        loss.backward()      # gradient 계산
        optimizer.step()     # gradient를 사용하여 파라미터를 업데이트

        loss_list.append(loss.item())
        """
        if iter % 20 == 0:
                elapsed_time = time.time() - start_t
                print('TRAIN(Elapsed: %fs): EPOCH %d/%d | BATCH %d/%d | LOSS: %f' %
                    (elapsed_time, epoch+1, num_epochs, iter+1, num_train, np.mean(loss_arr)))
                start_t = time.time() */ """
    
    print("Epoch: %d/%d | Loss: %.4f" % (epoch+1, num_epoch+1, np.mean(loss_list)))
    scheduler.step()
    
