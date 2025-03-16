import os
import av
import math
import random
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from info_nce import InfoNCE
from model import EVLTransformer

mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073])
std = torch.Tensor([0.26862954, 0.26130258, 0.27577711])
spatial_size = 224
num_frames = 16
sampling_rate = 16
num_temporal_views = 1
num_spatial_views = 1
decoder_num_layers = 8
decoder_qkv_dim = 1024
decoder_num_heads = 16

class VideoAudioDataset(Dataset):
    def __init__(
        self, video_data, audio_data, video_dir, audio_embed,
        num_spatial_views, num_temporal_views, num_frames, sampling_rate, spatial_size,
        mean, std
    ):
        self.video_data = video_data
        self.audio_data = audio_data
        self.video_dir = video_dir
        self.audio_embed = audio_embed

        self.spatial_size = spatial_size
        self.mean, self.std = mean, std
        self.num_frames, self.sampling_rate = num_frames, sampling_rate

        self.num_temporal_views = num_temporal_views
        self.num_spatial_views = num_spatial_views

    def __len__(self):
        return len(self.video_data)
    
    def __getitem__(self, idx):
        video_idx = self.video_data[idx]
        audio_idx = self.audio_data[idx]

        audio = self.audio_embed[audio_idx]

        dir = sorted(os.listdir(self.video_dir))[video_idx[0]]
        file = sorted(os.listdir(os.path.join(self.video_dir, dir)))[video_idx[1]]
        video_file = os.path.join(self.video_dir, dir, file)

        container = av.open(video_file)
        frames = {}
        for frame in container.decode(video=0):
            frames[frame.pts] = frame
        container.close()
        frames = [frames[k] for k in sorted(frames.keys())]
        frame_idx = []
        if (len(frames) != 0):
            for i in range(self.num_frames):
                frame_idx.append(i * self.sampling_rate if i * self.sampling_rate < len(frames) else frame_idx[-1])

        cropped_frames = []
        for x in frame_idx:
            img = frames[x].to_image()  # PIL image
            width, height = img.size    # Get dimensions

            new_size = min(width, height)
            left = (width - new_size) // 2
            top = (height - new_size) // 2
            right = left + new_size
            bottom = top + new_size
            img = img.crop((left, top, right, bottom))  # Crop the center of the image

            cropped_frame = av.video.frame.VideoFrame.from_image(img).reformat(width=self.spatial_size, height=self.spatial_size).to_rgb().to_ndarray()
            cropped_frames.append(cropped_frame)

        frames = cropped_frames
        if (len(frames) != 0):
            frames = torch.as_tensor(np.stack(frames)).float() / 255.
            frames = (frames - self.mean) / self.std
            frames = frames.permute(3, 0, 1, 2) # C, T, H, W
        else:
            frames = torch.as_tensor(np.array(frames))

        return frames, audio


def collate_fn(batch):
    # Filter out samples where `frames` is an empty tensor
    batch = [sample for sample in batch if sample[0].nelement() > 0]

    # If batch is not empty after filtering
    if len(batch) > 0:
        return torch.utils.data.dataloader.default_collate(batch)
    else:
        # Return None or handle empty batch case
        return None
    

def collectTrainData(pos_train, count_train, video_dir, train_audio_embeds, config):
    train_video_data = []
    train_audio_data = []

    for i in range(config['update']):
        li = []
        for k in range(count_train):
            li.append(k+1)
        for j in range(config['batch_size']):
            id = random.randint(0, len(li)-1)
            idx = li[id]
            audio = random.randint(pos_train[idx-1], pos_train[idx]-1)
            # video = (idx-1, random.randint(pos_train[idx-1], pos_train[idx]-1) - pos_train[idx-1])
            video = (idx-1, audio - pos_train[idx-1])
            train_audio_data.append(audio)
            train_video_data.append(video)
            del li[id]

    train_dataset = VideoAudioDataset(video_data=train_video_data, audio_data=train_audio_data,
                               video_dir=video_dir, audio_embed=train_audio_embeds,
                               num_spatial_views=num_spatial_views, num_temporal_views=num_temporal_views, 
                               num_frames=num_frames, sampling_rate=sampling_rate, 
                               spatial_size=spatial_size, mean=mean, std=std)
    
    # sampler = DistributedSampler(train_dataset, shuffle=False)
    train_dataloader = DataLoader(train_dataset, prefetch_factor=2,
                            batch_size=config['batch_size'], shuffle=False,
                            pin_memory=False, num_workers=4, collate_fn=collate_fn)
    
    return train_dataloader


def collectValidData(pos_valid, count_train, count_valid, video_dir, valid_audio_embeds, config):
    valid_video_data = []
    valid_audio_data = []

    for i in range(config['update'] // 10):
        li = []
        for k in range(count_valid):
            li.append(k+1)
        for j in range(config['batch_size']):
            id = random.randint(0, len(li)-1)
            idx = li[id]
            audio = random.randint(pos_valid[idx-1], pos_valid[idx]-1)
            video = (idx-1+count_train, audio - pos_valid[idx-1])
            valid_audio_data.append(audio)
            valid_video_data.append(video)
            del li[id]   

    valid_dataset = VideoAudioDataset(video_data=valid_video_data, audio_data=valid_audio_data,
                               video_dir=video_dir, audio_embed=valid_audio_embeds,
                               num_spatial_views=num_spatial_views, num_temporal_views=num_temporal_views, 
                               num_frames=num_frames, sampling_rate=sampling_rate, 
                               spatial_size=spatial_size, mean=mean, std=std)
    
    # sampler = DistributedSampler(valid_dataset, shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, prefetch_factor=2,
                            batch_size=config['batch_size'], shuffle=False,
                            pin_memory=False, num_workers=4, collate_fn=collate_fn)
    
    return valid_dataloader


def trainer(train_dataloader, valid_dataloader, model, optimizer, 
            criterion, config, epoch, device):

    ## training
    model.train()
    loss_record = []

    train_pbar = tqdm(train_dataloader, position=0, leave=True)
    optimizer.zero_grad()

    for i, (frames, audio) in enumerate(train_pbar):

        frames, audio = frames.to(device), audio.to(device)

        output = model(frames)
        loss = criterion(output, audio)
        iter_loss = loss.item()
        loss_record.append(iter_loss)
        loss = loss / config['accumulated_step']
        loss.backward()

        if ((i + 1) % config['accumulated_step'] == 0):
            optimizer.step()
            optimizer.zero_grad()
            
        # Display current epoch number and loss on tqdm progress bar.
        train_pbar.set_description(f'Train Epoch [{epoch+1}/{config["n_epoch"]}]')
        train_pbar.set_postfix({'loss': iter_loss})

    mean_train_loss = sum(loss_record)/len(loss_record)
    # scheduler.step(mean_train_loss)

    ## validate
    model.eval()
    loss_record = []

    valid_pbar = tqdm(valid_dataloader, position=0, leave=True)
    for i, (frames, audio) in enumerate(valid_pbar):
        frames, audio = frames.to(device), audio.to(device)
        with torch.no_grad():
            output = model(frames)
            loss = criterion(output, audio)

        loss_record.append(loss.item())

        valid_pbar.set_description(f'Valid Epoch [{epoch+1}/{config["n_epoch"]}]')
        valid_pbar.set_postfix({'loss': loss.detach().item()})

    mean_valid_loss = sum(loss_record) / len(loss_record)

    mean_valid_loss_tensor = torch.tensor(mean_valid_loss).to(device)
    # dist.all_reduce(mean_valid_loss_tensor, op=dist.ReduceOp.SUM)
    # mean_valid_loss = mean_valid_loss_tensor.item() / dist.get_world_size()
    mean_valid_loss = mean_valid_loss_tensor.item()

    print(f'Epoch [{epoch+1}/{config["n_epoch"]}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')

    # Free up memory (especially useful when training on GPU)
    torch.cuda.empty_cache()

    return mean_valid_loss


def main():

    # distributed training
    """
    dist.init_process_group(
        backend='nccl', 
        init_method='env://', 
        rank = torch.cuda.device_count(), 
        world_size = 1
    )
    dist.barrier()
    world_size = dist.get_world_size()
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    """

    pos_train = [0]
    pos_valid = [0]
    split = 0.9
    video_dir = './videos'
    dirs = sorted(os.listdir(video_dir))
    for i in range(1500):
        pos_train.append(pos_train[-1] + len(os.listdir(os.path.join(video_dir, dirs[i]))))
    for i in range(1500, 1550):
        pos_valid.append(pos_valid[-1] + len(os.listdir(os.path.join(video_dir, dirs[i]))))

    count_train = len(pos_train) - 1
    count_valid = len(pos_valid) - 1
    total_train = pos_train[-1]
    total_valid = pos_valid[-1]

    print(f"total train: {total_train}")
    print(f"total valid: {total_valid}")

    audio_train_file = './embeddings/train_audio.npy'
    train_audio_embeds = torch.from_numpy(np.asarray(np.memmap(audio_train_file, dtype='float32', mode='r+', shape=(total_train, 512))))
    audio_valid_file = './embeddings/valid_audio.npy'
    valid_audio_embeds = torch.from_numpy(np.asarray(np.memmap(audio_valid_file, dtype='float32', mode='r+', shape=(total_valid, 512))))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    config = {
        'n_epoch': 50,
        'update': 20,
        'batch_size': 16,
        'accumulated_step': 8,
        'learning_rate': 1e-3,
        'save_path': './model.pt'
    }

    model = EVLTransformer(
        num_frames=num_frames,
        backbone_name="ViT-L/14-lnpre",
        backbone_type="clip",
        backbone_path="./checkpoint/ViT-L-14.pt",
        backbone_mode="freeze_fp16",
        decoder_num_layers=decoder_num_layers,
        decoder_qkv_dim=decoder_qkv_dim,
        decoder_num_heads=decoder_num_heads,
        num_classes=512
    )
    model.to(device)
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    n_epochs, best_loss = config['n_epoch'], math.inf
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-6)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    criterion = InfoNCE(temperature=0.1)

    for epoch in range(n_epochs):
        
        train_dataloader = collectTrainData(pos_train=pos_train, count_train=count_train,
                                            video_dir=video_dir, train_audio_embeds=train_audio_embeds,
                                            config=config)
        valid_dataloader = collectValidData(pos_valid=pos_valid, count_train=count_train, count_valid=count_valid,
                                            video_dir=video_dir, valid_audio_embeds=valid_audio_embeds, 
                                            config=config)

        # train for one epoch
        mean_valid_loss = trainer(train_dataloader, valid_dataloader, model, optimizer, 
                                criterion, config, epoch, device)
        
        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path']) # Save your best model
            print('Saving model with loss {:.3f}...'.format(best_loss))

if __name__ == '__main__':
    main()