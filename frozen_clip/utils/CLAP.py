import os
import os.path as path
import numpy as np
import torch
import torchaudio
import laion_clap
from tqdm import tqdm
from torch_audiomentations import (
    Compose, Gain, PolarityInversion, 
    PitchShift, SpliceOut, PeakNormalization
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel= 'HTSAT-base', device=device)
clap_model.load_ckpt("../checkpoint/music_audioset_epoch_15_esc_90.14.pt")

fixed_sr = 48000
audio_seg_length = 8
audio_path = "../clipclap"
test_path = "../test"

"""
apply_augmentation = Compose(
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
"""
apply_augmentation = Compose(
    transforms=[PeakNormalization()]
)

## --- training part --- ##
train_dir = os.path.join(audio_path, 'train', 'audios')
dirs = sorted(os.listdir(train_dir))
for i in tqdm(range(len(dirs))):
    audio_files = sorted(os.listdir(os.path.join(train_dir, dirs[i])))
    for j in range(len(audio_files)):
        data, sr = torchaudio.load(os.path.join(train_dir, dirs[i], audio_files[j]))
        # data = data.to(device)
        data = torchaudio.functional.resample(data, orig_freq=sr, new_freq=fixed_sr)
        data = data[:, :int(fixed_sr*audio_seg_length)]
        # print(data.shape)
        data = data.unsqueeze(0)   
        aug_data = apply_augmentation(data, sample_rate=fixed_sr)
        # output data
        # torchaudio.save(os.path.join(test_path, f"{dirs[i]}_{audio_files[j].split('.')[0]}.mp3"), aug_data.squeeze(0).cpu(), fixed_sr)
        if aug_data.shape[0] > 1:
            aug_data = torch.mean(aug_data, dim=0, keepdim=True)
        aug_data = aug_data.reshape(1, -1).cpu().numpy()
        if (j == 0):
            audios = aug_data
        else:
            audios = np.concatenate((audios, aug_data), axis=0)

    audio_embed = clap_model.get_audio_embedding_from_data(x=audios, use_tensor=False)
    if (i == 0):
        train_audio_embeds = audio_embed
    else:
        train_audio_embeds = np.concatenate((train_audio_embeds, audio_embed), axis=0)

print(train_audio_embeds.shape)
filename1 = '../embeddings/train_audio.npy'
if (os.path.isfile(filename1)):
    os.remove(filename1)
os.system("touch {}".format(filename1))
fp1 = np.memmap(filename1, dtype='float16', mode='w+', shape=(train_audio_embeds.shape[0], train_audio_embeds.shape[1]))
fp1[:] = train_audio_embeds[:]
fp1.filename == path.abspath(filename1)
fp1.flush()


## --- validating part --- ##
valid_dir = os.path.join(audio_path, 'valid', 'audios')
dirs = sorted(os.listdir(valid_dir))
for i in tqdm(range(len(dirs))):
    audio_files = sorted(os.listdir(os.path.join(valid_dir, dirs[i])))
    for j in range(len(audio_files)):
        data, sr = torchaudio.load(os.path.join(valid_dir, dirs[i], audio_files[j]))
        # data = data.to(device)
        data = torchaudio.functional.resample(data, orig_freq=sr, new_freq=fixed_sr)
        data = data[:, :int(fixed_sr*audio_seg_length)]
        # print(data.shape)
        data = data.unsqueeze(0)   
        aug_data = apply_augmentation(data, sample_rate=fixed_sr)
        # output data
        # torchaudio.save(os.path.join(test_path, f"{dirs[i]}_{audio_files[j].split('.')[0]}.mp3"), aug_data.squeeze(0).cpu(), fixed_sr)
        if aug_data.shape[0] > 1:
            aug_data = torch.mean(aug_data, dim=0, keepdim=True)
        aug_data = aug_data.reshape(1, -1).cpu().numpy()
        if (j == 0):
            audios = aug_data
        else:
            audios = np.concatenate((audios, aug_data), axis=0)

    audio_embed = clap_model.get_audio_embedding_from_data(x=audios, use_tensor=False)
    if (i == 0):
        valid_audio_embeds = audio_embed
    else:
        valid_audio_embeds = np.concatenate((valid_audio_embeds, audio_embed), axis=0)

print(valid_audio_embeds.shape)
filename2 = '../embeddings/valid_audio.npy'
if (os.path.isfile(filename2)):
    os.remove(filename2)
os.system("touch {}".format(filename2))
fp2 = np.memmap(filename2, dtype='float16', mode='w+', shape=(valid_audio_embeds.shape[0], valid_audio_embeds.shape[1]))
fp2[:] = valid_audio_embeds[:]
fp2.filename == path.abspath(filename2)
fp2.flush()