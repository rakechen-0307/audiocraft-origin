import os
import av
import math
import random
import torch
import torchaudio
import numpy as np
from frozen_clip.model import EVLTransformer
from audiocraft.models.musicgen import MusicGenCLAP
from audiocraft.data.audio import audio_write
from audiocraft.data.audio_utils import convert_audio

seg_length = 10
sample_rate = 48000
mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073])
std = torch.Tensor([0.26862954, 0.26130258, 0.27577711])
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def _get_param_spatial_crop(
    scale, ratio, height, width, num_repeat=10, log_scale=True, switch_hw=False
):
    """
    Given scale, ratio, height and width, return sampled coordinates of the videos.
    """
    for _ in range(num_repeat):
        area = height * width
        target_area = random.uniform(*scale) * area
        if log_scale:
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))
        else:
            aspect_ratio = random.uniform(*ratio)

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if np.random.uniform() < 0.5 and switch_hw:
            w, h = h, w

        if 0 < w <= width and 0 < h <= height:
            i = random.randint(0, height - h)
            j = random.randint(0, width - w)
            return i, j, h, w

    # Fallback to central crop
    in_ratio = float(width) / float(height)
    if in_ratio < min(ratio):
        w = width
        h = int(round(w / min(ratio)))
    elif in_ratio > max(ratio):
        h = height
        w = int(round(h * max(ratio)))
    else:  # whole image
        w = width
        h = height
    i = (height - h) // 2
    j = (width - w) // 2
    return i, j, h, w

def random_resized_crop(
    images,
    target_height,
    target_width,
    scale=(0.08, 1.0),
    ratio=(3.0 / 4.0, 4.0 / 3.0),
):
    """
    Crop the given images to random size and aspect ratio. A crop of random
    size (default: of 0.08 to 1.0) of the original size and a random aspect
    ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This
    crop is finally resized to given size. This is popularly used to train the
    Inception networks.

    Args:
        images: Images to perform resizing and cropping.
        target_height: Desired height after cropping.
        target_width: Desired width after cropping.
        scale: Scale range of Inception-style area based random resizing.
        ratio: Aspect ratio range of Inception-style area based random resizing.
    """

    height = images.shape[2]
    width = images.shape[3]

    i, j, h, w = _get_param_spatial_crop(scale, ratio, height, width)
    cropped = images[:, :, i : i + h, j : j + w]
    return torch.nn.functional.interpolate(
        cropped,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )


def sample_frame_idx(len):
    frame_indices = []

    if 16 < 0: # tsn sample
        seg_size = (len - 1) / 16
        for i in range(16):
            frame_indices.append(round(seg_size * i))
    elif 16 * (16 - 1) + 1 >= len:
        for i in range(16):
            frame_indices.append(i * 16 if i * 16 < len else frame_indices[-1])
    else:
        frame_indices = list(range(0, 0 + 16 * 16, 16))

    return frame_indices

def video_to_frame(video_file):
    container = av.open(video_file)
    frames = {}
    for frame in container.decode(video=0):
        frames[frame.pts] = frame
    container.close()
    frames = [frames[k] for k in sorted(frames.keys())]
    frame_idx = sample_frame_idx(len(frames))
    frames = [frames[x].to_rgb().to_ndarray() for x in frame_idx]
    frames = torch.as_tensor(np.stack(frames)).float() / 255.

    frames = (frames - mean) / std
    frames = frames.permute(3, 0, 1, 2) # C, T, H, W
    frames = random_resized_crop(
        frames, 224, 224,
    )
    return frames


music_model = MusicGenCLAP.get_pretrained('checkpoints/clapemb(spotify-small-80)')
music_model.set_generation_params(duration=30, cfg_coef=3.0)
clap_conditioner = music_model.lm.condition_provider.conditioners.description
print(clap_conditioner)

clipclap_model = EVLTransformer(
    num_frames=16,
    backbone_name='ViT-L/14-lnpre',
    backbone_type='clip',
    backbone_mode='freeze_fp16',
    backbone_path='./frozen_clip/checkpoint/ViT-L-14.pt',
    decoder_num_layers=4,
    decoder_qkv_dim=1024,
    decoder_num_heads=16,
    num_classes=512
)
clipclap_model.to(device)
state_dict = torch.load("./frozen_clip/checkpoint/best.pt", map_location='cpu')
msg = clipclap_model.load_state_dict(state_dict, strict=False)
# print(msg)

sample_dir = "/work/u2614323/code/audiocraft-origin/samples/videos"
sample_files = sorted(os.listdir(sample_dir))
files = []
file_names = []
for i in range(len(sample_files)):
    files.append(sample_dir + "/" + sample_files[i])
    file_names.append(sample_files[i])

audio = []
for i in range(len(files)):
    video_file = files[i]
    frames = video_to_frame(video_file)
    frames = frames.unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(True):
            audio_embed = clipclap_model(frames)
    
    audio_embed = audio_embed.cpu().unsqueeze(0)
    null_condition = torch.zeros(1, sample_rate*seg_length)
    null_embed = clap_conditioner.clap.get_audio_embedding_from_data(null_condition, use_tensor=True).unsqueeze(0)

    embed = torch.cat((audio_embed, null_embed), dim=0)
    B = embed.shape[0]
    out_embed = clap_conditioner.output_proj(embed).view(B, -1, clap_conditioner.output_dim)

    if clap_conditioner.normalize:
        out_embed = torch.nn.functional.normalize(out_embed, p=2.0, dim=-1)

    

"""
for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f"{file_names[idx].split('.')[0]}", one_wav.cpu(), music_model.sample_rate, strategy="loudness", loudness_compressor=True)
"""