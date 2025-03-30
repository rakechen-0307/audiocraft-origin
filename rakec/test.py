import os
import av
import math
import random
import torch
import torchaudio
import numpy as np
import laion_clap
import librosa
import subprocess
from frozen_clip.model import EVLTransformer
from audiocraft.models.musicgen import MusicGenCLAP
from audiocraft.data.audio import audio_write
from audiocraft.data.audio_utils import convert_audio

seg_length = 10
sample_rate = 48000
num_frames = 16
sampling_rate = -1
mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073])
std = torch.Tensor([0.26862954, 0.26130258, 0.27577711])
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def combine_audio_video(video_path, audio_path, output_path):
    cmd = [
        'ffmpeg',
        '-i', video_path,        # Input video
        '-i', audio_path,        # Input audio
        '-c:v', 'copy',          # Copy the video stream without re-encoding
        '-c:a', 'aac',           # Encode audio as AAC (common for MP4)
        '-map', '0:v',           # Use video from first input
        '-map', '1:a',           # Use audio from second input
        '-shortest',             # End when the shortest input ends
        '-y',                    # Overwrite output files without asking
        output_path              # Output file
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully combined {video_path} and {audio_path} into {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error combining files: {e}")

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

    if sample_rate < 0: # tsn sample
        seg_size = (len - 1) / num_frames
        for i in range(num_frames):
            frame_indices.append(round(seg_size * i))
    elif sample_rate * (num_frames - 1) + 1 >= len:
        for i in range(num_frames):
            frame_indices.append(i * sampling_rate if i * sampling_rate < len else frame_indices[-1])
    else:
        frame_indices = list(range(0, 0 + sampling_rate * num_frames, sampling_rate))

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


ma = "HTSAT-base"
clap_model = laion_clap.CLAP_Module(enable_fusion = False, amodel = ma)
clap_model.load_ckpt("./rakec/clap/music_audioset_epoch_15_esc_90.14.pt")
clap_model.eval()

music_model = MusicGenCLAP.get_pretrained('checkpoints/clapemb(spotify-small-new)')
music_model.set_generation_params(duration=10, cfg_coef=3.0)
clap_conditioner = music_model.lm.condition_provider.conditioners.description

clipclap_model = EVLTransformer(
    num_frames=num_frames,
    backbone_name='ViT-L/14-lnpre',
    backbone_type='clip',
    backbone_mode='freeze_fp16',
    backbone_path='./frozen_clip/checkpoints/ViT-L-14.pt',
    decoder_num_layers=4,
    decoder_qkv_dim=1024,
    decoder_num_heads=16,
    num_classes=512
)
clipclap_model.to(device)
clipclap_model.eval()
state_dict = torch.load("./frozen_clip/checkpoints/vitl14-16f-first.pt", map_location='cpu')
msg = clipclap_model.load_state_dict(state_dict)

audio_dir = "./samples/audios"
sample_dir = "./samples/videos"
sample_files = sorted(os.listdir(sample_dir))
files = []
file_names = []
audio_files = []
for i in range(len(sample_files)):
    files.append(sample_dir + "/" + sample_files[i])
    file_names.append(sample_files[i])
    audio_files.append(audio_dir + "/" + f"{sample_files[i].split('.')[0]}.mp3")

audio = []
avg_sim = []
for i in range(len(files)):
    video_file = files[i]
    frames = video_to_frame(video_file)
    frames = frames.unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(True):
            audio_embed = clipclap_model(frames)
    
    with torch.no_grad():
        print(audio_files[i])
        audio_data, _ = librosa.load(audio_files[i], sr=48000)
        audio_data = audio_data.reshape(1, -1)
        if (audio_data.shape[-1] >= 480000):
            audio_data = audio_data[:, :480000]
        clap_embed = clap_model.get_audio_embedding_from_data(x = audio_data, use_tensor=False)

    cosine_similarity = np.dot(audio_embed.cpu().numpy(), clap_embed.T) / (np.linalg.norm(audio_embed.cpu().numpy()) * np.linalg.norm(clap_embed))
    print(f"similarity: {cosine_similarity[0][0]}")
    avg_sim.append(cosine_similarity[0][0])

    audio_embed = audio_embed.unsqueeze(0)
    null_condition = torch.zeros(1, sample_rate*seg_length)
    null_embed = clap_conditioner.clap.get_audio_embedding_from_data(null_condition, use_tensor=True).unsqueeze(0)

    embed = torch.cat((audio_embed, null_embed), dim=0)
    empty_idx = torch.LongTensor([1])
    B = embed.shape[0]
    out_embed = clap_conditioner.output_proj(embed).view(B, -1, clap_conditioner.output_dim)

    if clap_conditioner.normalize:
        out_embed = torch.nn.functional.normalize(out_embed, p=2.0, dim=-1)

    mask = torch.ones(*out_embed.shape[:2], device=out_embed.device)
    mask[empty_idx, :] = 0  # zero-out index where the input is non-existant
    out_embed = (out_embed * mask.unsqueeze(-1))

    cfg_conditions = {
        'description': (
            out_embed, mask
        )
    }

    wav = music_model.generate_with_conditions(cfg_conditions)
    audio_write(f"{file_names[i].split('.')[0]}", wav.cpu().squeeze(0), music_model.sample_rate, strategy="loudness", loudness_compressor=True)
    combine_audio_video(video_file, f"{file_names[i].split('.')[0]}.wav", f"{file_names[i].split('.')[0]}.mp4")

print(f"avg similarity: {sum(avg_sim)/len(avg_sim)}")