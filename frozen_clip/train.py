import av
import os
import copy
import math
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import laion_clap
import torch
import torchaudio
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from collections import OrderedDict
from info_nce import InfoNCE
from model import EVLTransformer
from weight_loaders import weight_loader_fn_dict
from vision_transformer import vit_presets
from transform import create_random_augment, random_resized_crop
from torch_audiomentations import (
    Compose, Gain, PolarityInversion, 
    PitchShift, SpliceOut, PeakNormalization
)

## parameters
def config_args():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--num_epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--update_per_epoch', type=int, default=1, help='update per epoch')
    parser.add_argument('--batch_split', type=int, default=16, help='optionally split the batch into smaller shards and forward/backward one shard at a time to avoid out-of-memory error.')
    parser.add_argument('--save_freq', type=int, default=10, help='save a checkpoint every N epochs')
    parser.add_argument('--eval_freq', type=int, default=1, help='evaluate every N epochs')
    parser.add_argument('--backbone_name', type=str, choices=vit_presets.keys(), default='ViT-L/14-lnpre',
                        help='the backbone variant used to generate image feature maps')
    parser.add_argument('--backbone_path', type=str, help='path to pretrained backbone weights')
    parser.add_argument('--backbone_type', type=str, default='clip', choices=weight_loader_fn_dict.keys(),
                        help='type of backbone weights (used to determine how to convert state_dict from different pretraining codebase)')
    parser.add_argument('--finetune_backbone', action='store_true', help='finetune backbone weights')
    parser.add_argument('--decoder_num_layers', type=int, default=4, help='number of decoder layers')
    parser.add_argument('--decoder_qkv_dim', type=int, default=1024, help='q (k, v) projection output dimensions in decoder attention layers')
    parser.add_argument('--decoder_num_heads', type=int, default=16, help='number of heads in decoder attention layers')
    parser.add_argument('--decoder_mlp_factor', type=float, default=4.0, help='expansion factor of feature dimension in the middle of decoder MLPs')
    parser.add_argument('--num_classes', type=int, default=512, help='number of classes')
    parser.add_argument('--cls_dropout', type=float, default=0.2, help='dropout rate applied before the final classification linear projection')
    parser.add_argument('--decoder_mlp_dropout', type=float, default=0.2, help='dropout rate applied in MLP layers in the decoder')
    parser.add_argument('--no_temporal_conv', action='store_false', dest='temporal_conv', help='disable temporal convolution on frame features')
    parser.add_argument('--no_temporal_pos_embed', action='store_false', dest='temporal_pos_embed', help='disable temporal position embeddings added to frame features')
    parser.add_argument('--no_temporal_cross_attention', action='store_false', dest='temporal_cross_attention', help='disable temporal cross attention on frame query and key features')
    parser.set_defaults(temporal_conv=True, temporal_pos_embed=True, temporal_cross_attention=True)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--min_lr', type=float, default=5e-6, help='minimum learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='optimizer weight decay')
    parser.add_argument('--disable_fp16', action='store_false', dest='fp16', help='disable fp16 during training or inference')
    parser.set_defaults(fp16=True)
    # dataset
    parser.add_argument('--data_root', type=str, help='data folder')
    parser.add_argument('--list_txt', type=str, help='list of information')
    parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
    # parser.add_argument('--num_spatial_views', type=int, default=1, help='number of spatial crops used for testing (total views = num_spatial_views * num_temporal_views)')
    # parser.add_argument('--num_temporal_views', type=int, default=1, help='number of temporal crops used for testing (total views = num_spatial_views * num_temporal_views)')
    parser.add_argument('--num_frames', type=int, default=16, help='number of frames used for each view')
    parser.add_argument('--sampling_rate', type=int, default=16, help='temporal stride for frame sampling, only valid when tsn_sampling is not enabled')
    parser.add_argument('--tsn_sampling', action='store_true', help='enable TSN-style sampling (i.e. sample frames with dynamic stride to cover the whole video)')
    parser.add_argument('--spatial_size', type=int, default=224, help='frame height and width in pixels')
    parser.add_argument('--mean', type=float, nargs='+', help='pixel mean used to normalize the image.')
    parser.add_argument('--std', type=float, nargs='+', help='pixel std used to normalize the image')
    parser.add_argument('--num_workers', type=int, default=4, help='number of DataLoader worker threads')
    parser.add_argument('--dummy_dataset', action='store_true', help='use fake datasets that generate all 0 (use for speed test only)')
    parser.add_argument('--auto_augment', type=str, default='rand-m7-n4-mstd0.5-inc1', help='auto augment configuration')
    parser.add_argument('--interpolation', type=str, default='bicubic', help='interpolation mode')
    # pretrained
    parser.add_argument('--resume_path', type=str, default=None, help='resume from manually specified checkpoint file, overriding auto_resume')
    parser.add_argument('--load_proj', action='store_true', help='whether to load projection layer checkpoint')

    return parser

class VideoAudioDataset(Dataset):
    def __init__ (
        self, video_idxes, audio_idxes, video_dir, audio_dir, num_frames,
        sampling_rate, spatial_size, mean, std, clap_model, audio_augmentation = None, 
        auto_augment = None, interpolation = 'bicubic', random_sample = False, 
        audio_sampling_rate = 48000, audio_seg_length = 8
    ):
        self.video_idxes = video_idxes
        self.audio_idxes = audio_idxes
        self.video_dir = video_dir
        self.audio_dir = audio_dir

        self.spatial_size = spatial_size
        self.mean, self.std = mean, std
        self.num_frames, self.sampling_rate = num_frames, sampling_rate
        self.interpolation = interpolation
        self.auto_augment = auto_augment
        self.random_sample = random_sample

        self.clap_model = clap_model
        self.audio_augmentation = audio_augmentation
        self.audio_sampling_rate = audio_sampling_rate
        self.audio_seg_length = audio_seg_length
    
    def __len__(self):
        return len(self.video_idxes)

    def __getitem__(self, idx):
        video_idx = self.video_idxes[idx]
        audio_idx = self.audio_idxes[idx]

        audio_file = os.path.join(self.audio_dir, audio_idx[0], audio_idx[1])
        audio_data, sr = torchaudio.load(audio_file)
        audio_data = torchaudio.functional.resample(audio_data, orig_freq=sr, new_freq=self.audio_sampling_rate)
        audio_data = audio_data[:, :int(self.audio_sampling_rate*self.audio_seg_length)]
        if (self.audio_augmentation is not None):
            audio_data = audio_data.unsqueeze(0)   
            audio_data = self.audio_augmentation(audio_data, sample_rate=self.audio_sampling_rate).squeeze(0)
        if audio_data.shape[0] > 1:
            audio_data = torch.mean(audio_data, dim=0, keepdim=True)
        audio_data = audio_data.reshape(1, -1).cpu().numpy()
        audio = self.clap_model.get_audio_embedding_from_data(x=audio_data, use_tensor=False).squeeze(0)

        video_file = os.path.join(self.video_dir, video_idx[0], video_idx[1])
        container = av.open(video_file)
        frames = {}
        for frame in container.decode(video=0):
            frames[frame.pts] = frame
        container.close()
        frames = [frames[k] for k in sorted(frames.keys())]

        if (self.random_sample):
            frame_idx = self._random_sample_frame_idx(len(frames))
        else:
            frame_idx = self._sample_frame_idx(len(frames))
        frames = [frames[x].to_rgb().to_ndarray() for x in frame_idx]
        frames = torch.as_tensor(np.stack(frames)).float() / 255.

        if self.auto_augment is not None:
            aug_transform = create_random_augment(
                input_size=(frames.size(1), frames.size(2)),
                auto_augment=self.auto_augment,
                interpolation=self.interpolation,
            )
            frames = frames.permute(0, 3, 1, 2) # T, C, H, W
            frames = [transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))]
            frames = aug_transform(frames)
            frames = torch.stack([transforms.ToTensor()(img) for img in frames])
            frames = frames.permute(0, 2, 3, 1)

        frames = (frames - self.mean) / self.std
        frames = frames.permute(3, 0, 1, 2) # C, T, H, W
        frames = random_resized_crop(
            frames, self.spatial_size, self.spatial_size,
        )

        return frames, audio

    def _random_sample_frame_idx(self, len):
        frame_indices = []

        if self.sampling_rate < 0: # tsn sample
            seg_size = (len - 1) / self.num_frames
            for i in range(self.num_frames):
                start, end = round(seg_size * i), round(seg_size * (i + 1))
                frame_indices.append(np.random.randint(start, end + 1))
        elif self.sampling_rate * (self.num_frames - 1) + 1 >= len:
            for i in range(self.num_frames):
                frame_indices.append(i * self.sampling_rate if i * self.sampling_rate < len else frame_indices[-1])
        else:
            start = np.random.randint(len - self.sampling_rate * (self.num_frames - 1))
            frame_indices = list(range(start, start + self.sampling_rate * self.num_frames, self.sampling_rate))

        return frame_indices
    
    def _sample_frame_idx(self, len):
        frame_indices = []

        if self.sampling_rate < 0: # tsn sample
            seg_size = (len - 1) / self.num_frames
            for i in range(self.num_frames):
                frame_indices.append(round(seg_size * i))
        elif self.sampling_rate * (self.num_frames - 1) + 1 >= len:
            for i in range(self.num_frames):
                frame_indices.append(i * self.sampling_rate if i * self.sampling_rate < len else frame_indices[-1])
        else:
            frame_indices = list(range(0, 0 + self.sampling_rate * self.num_frames, self.sampling_rate))

        return frame_indices

def collectTrainData(
    categories_list, len_categories, train_video_dir, train_audio_dir, 
    mean, std, clap_model, audio_augmentation, args
):
    video_idxes = []
    audio_idxes = []

    for i in range(args.update_per_epoch):
        copy_categories_list = copy.deepcopy(categories_list)
        count = 0
        for j in range(args.batch_size):
            category = copy_categories_list[count]
            dir_idx = random.randint(0, len(category)-1)
            dir_id = category[dir_idx]
            del copy_categories_list[count][dir_idx]
            file_idx = random.randint(0, len(sorted(os.listdir(os.path.join(train_video_dir, dir_id))))-1)
            audio = (dir_id, f"{file_idx:03d}.mp3")
            video = (dir_id, f"{file_idx:03d}.mp4")
            audio_idxes.append(audio)
            video_idxes.append(video)

            count += 1
            if (count == len_categories):
                count = 0

    train_dataset = VideoAudioDataset(
        video_idxes=video_idxes, audio_idxes=audio_idxes, video_dir=train_video_dir,
        audio_dir=train_audio_dir, num_frames=args.num_frames, 
        sampling_rate=-1 if args.tsn_sampling else args.sampling_rate, 
        spatial_size=args.spatial_size, mean=mean, std=std, interpolation=args.interpolation,
        auto_augment=args.auto_augment, random_sample=True, clap_model=clap_model,
        audio_augmentation=audio_augmentation
    )
    
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, pin_memory=False, drop_last=True
    )
    return train_dataloader

def collectValidData(
    count_valid, valid_video_dir, valid_audio_dir, 
    mean, std, clap_model, args
):
    video_idxes = []
    audio_idxes = []
    dirs = sorted(os.listdir(valid_video_dir))

    for i in range(count_valid):
        dir_id = dirs[i]
        file_idx = random.randint(0, len(sorted(os.listdir(os.path.join(valid_video_dir, dir_id))))-1)
        audio = (dir_id, f"{file_idx:03d}.mp3")
        video = (dir_id, f"{file_idx:03d}.mp4")
        audio_idxes.append(audio)
        video_idxes.append(video)

    valid_dataset = VideoAudioDataset(
        video_idxes=video_idxes, audio_idxes=audio_idxes, video_dir=valid_video_dir,
        audio_dir=valid_audio_dir, num_frames=args.num_frames, 
        sampling_rate=-1 if args.tsn_sampling else args.sampling_rate, 
        spatial_size=args.spatial_size, mean=mean, std=std, 
        interpolation=args.interpolation, auto_augment=None, random_sample=False,
        clap_model=clap_model
    )

    valid_dataloader = DataLoader(
        dataset=valid_dataset, batch_size=args.batch_size // args.batch_split, shuffle=False, 
        num_workers=args.num_workers, pin_memory=False, drop_last=False
    )
    return valid_dataloader

def trainer(
    train_dataloader, model, optimizer, criterion, scheduler, 
    loss_scaler, epoch, device, args
):
    model.train()
    loss_record = []
    print(f"Learning Rate of Epoch {epoch + 1}: {scheduler.get_last_lr()[0]}")
    train_pbar = tqdm(train_dataloader, position=0, leave=True)

    for i, batch in enumerate(train_pbar):
        frames, audio = batch
        optimizer.zero_grad()
        loss_value = 0
        split_size = frames.size(0) // args.batch_split
        for j in range(args.batch_split):
            frames_slice = frames[split_size * j: split_size * (j + 1)]
            audio_slice = audio[split_size * j: split_size * (j + 1)]
            frames_slice, audio_slice = frames_slice.to(device), audio_slice.to(device)

            with torch.cuda.amp.autocast(args.fp16):
                output = model(frames_slice)
                loss = criterion(output, audio_slice)

            loss_value += loss.item() / args.batch_split
            loss_scaler.scale(loss / args.batch_split).backward()

        loss_scaler.step(optimizer)
        loss_scaler.update()

        train_pbar.set_description(f'Train Epoch [{epoch + 1}/{args.num_epochs}]')
        train_pbar.set_postfix({'loss': loss_value})
        loss_record.append(loss_value)

    scheduler.step()  
    mean_train_loss = sum(loss_record) / len(loss_record)
        
    print(f'Epoch [{epoch + 1}/{args.num_epochs}]: Training Loss: {mean_train_loss:.4f}')  
    return mean_train_loss

def validate(valid_dataloader, model, criterion, epoch, device, args):
    model.eval()
    loss_record = []
    valid_pbar = tqdm(valid_dataloader, position=0, leave=True)

    for i, batch in enumerate(valid_pbar):
        frames, audio = batch
        frames, audio = frames.to(device), audio.to(device)
        with torch.no_grad():
            with torch.cuda.amp.autocast(args.fp16):
                output = model(frames)
                loss = criterion(output, audio)
        loss_record.append(loss.item())
    
    mean_valid_loss = sum(loss_record) / len(loss_record)      
    print(f'Epoch [{epoch + 1}/{args.num_epochs}]: Validation Loss: {mean_valid_loss:.4f}')

    return mean_valid_loss
        

def main():
    parser = config_args()
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    assert args.batch_size % args.batch_split == 0
    mean = torch.Tensor(args.mean)
    std = torch.Tensor(args.std)

    categories_list = []
    category = []
    id_file = open(args.list_txt, "r+")
    id_list = id_file.readlines()
    for id in id_list:
        id = id.replace(" ", "")
        id = id.replace("\n", "")
        if (id == ""):
            categories_list.append(category)
            category = []
        else:
            category.append(id)
    categories_list.append(category)
    len_categories = len(categories_list)

    clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    clap_model.load_ckpt("./checkpoint/music_audioset_epoch_15_esc_90.14.pt")
    fixed_sr = 48000
    audio_augmentation = Compose(
        transforms=[
            PeakNormalization(),
            Gain(min_gain_in_db=-6.0, max_gain_in_db=6.0, sample_rate=fixed_sr, target_rate=fixed_sr),
            PolarityInversion(sample_rate=fixed_sr, target_rate=fixed_sr),
            PitchShift(min_transpose_semitones=-12.0, max_transpose_semitones=12.0,
                    sample_rate=fixed_sr, target_rate=fixed_sr),
            SpliceOut(num_time_intervals=8, max_width=400, 
                    sample_rate=fixed_sr, target_rate=fixed_sr)
        ]
    )

    model = EVLTransformer(
        num_frames=args.num_frames,
        backbone_name=args.backbone_name,
        backbone_type=args.backbone_type,
        backbone_mode='finetune' if args.finetune_backbone else ('freeze_fp16' if args.fp16 else 'freeze_fp32'),
        backbone_path=args.backbone_path,
        decoder_num_layers=args.decoder_num_layers,
        decoder_qkv_dim=args.decoder_qkv_dim,
        decoder_num_heads=args.decoder_num_heads,
        num_classes=args.num_classes,
        cls_dropout=args.cls_dropout,
        decoder_mlp_dropout=args.decoder_mlp_dropout
    )
    model = model.to(device)

    # Initialize projection layer
    nn.init.ones_(model.proj[0].weight)
    nn.init.zeros_(model.proj[0].bias)
    nn.init.trunc_normal_(model.proj[2].weight, std=0.02)
    if model.proj[2].bias is not None:
        nn.init.zeros_(model.proj[2].bias)
            
    # Load checkpoint
    if args.resume_path:
        if args.load_proj:
            ckpt = torch.load(args.resume_path, map_location='cpu')
            pretrained_state_dict = OrderedDict(
                (key.replace("module.", "", 1), value) for key, value in ckpt.items()
            ) 
        else:
            ckpt = torch.load(args.resume_path, map_location='cpu')['model']
            pretrained_state_dict = OrderedDict(
                (key.replace("module.", "", 1), value) for key, value in ckpt.items() if "proj.2" not in key
            )
        msg = model.load_state_dict(pretrained_state_dict, strict=False)  
        print(f"Loading pretrained model: {msg}")

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.num_epochs,
        eta_min=args.min_lr
    )
    loss_scaler = torch.cuda.amp.grad_scaler.GradScaler(enabled=args.fp16)
    criterion = InfoNCE(temperature=0.05)

    ## load data
    train_dir = os.path.join(args.data_root, 'train')
    valid_dir = os.path.join(args.data_root, 'valid')
    train_video_dir = os.path.join(train_dir, 'videos')
    valid_video_dir = os.path.join(valid_dir, 'videos')
    train_audio_dir = os.path.join(train_dir, 'audios')
    valid_audio_dir = os.path.join(valid_dir, 'audios')

    count_train = len(sorted(os.listdir(train_video_dir)))
    count_valid = len(sorted(os.listdir(valid_video_dir)))
    print(f"train count: {count_train}")
    print(f"valid count: {count_valid}")

    ## training
    train_losses = []
    valid_losses = []
    best_valid_loss = math.inf
    for epoch in range(args.num_epochs):
        train_dataloader = collectTrainData(
            categories_list=categories_list, len_categories=len_categories, 
            train_video_dir=train_video_dir, train_audio_dir=train_audio_dir, 
            mean=mean, std=std, args=args, clap_model=clap_model, 
            audio_augmentation=audio_augmentation
        )
        valid_dataloader = collectValidData(
            count_valid=count_valid, valid_video_dir=valid_video_dir, 
            valid_audio_dir=valid_audio_dir, mean=mean, std=std, args=args,
            clap_model=clap_model
        )

        trainer(
            train_dataloader=train_dataloader, model=model, optimizer=optimizer,
            criterion=criterion, scheduler=scheduler, loss_scaler=loss_scaler, epoch=epoch,
            device=device, args=args
        )

        if ((epoch + 1) % args.eval_freq == 0):
            valid_loss = validate(
                valid_dataloader=valid_dataloader, model=model, criterion=criterion,
                epoch=epoch, device=device, args=args
            )
            if valid_loss < best_valid_loss:
                model_to_save = model
                torch.save(model_to_save.state_dict(), './best.pt')
                best_valid_loss = valid_loss
        
        if ((epoch + 1) % args.save_freq == 0):
            model_to_save = model
            torch.save(model_to_save.state_dict(), f'./{(epoch + 1):03d}.pt')

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')

    valid_x = list(range(0, len(train_losses), args.eval_freq))
    plt.plot(valid_x, valid_losses, label='Validation Loss')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('loss_curve.png')
    plt.close()

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()