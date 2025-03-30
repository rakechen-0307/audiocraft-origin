import os
import torch
import torchaudio
import numpy as np
import laion_clap

clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel= 'HTSAT-base')
clap_model.load_ckpt("./music_audioset_epoch_15_esc_90.14.pt")

gt_dir = "./samples/audios"
output_dir = "./wav/ours"
gt_files = sorted(os.listdir(gt_dir))
output_files = sorted(os.listdir(output_dir))
avg_sim = []
for i in range(len(gt_files)):
    gt_file = os.path.join(gt_dir, gt_files[i])
    output_file = os.path.join(output_dir, output_files[i])

    gt_data, sr = torchaudio.load(gt_file)
    gt_data = torchaudio.functional.resample(gt_data, orig_freq=sr, new_freq=48000)
    gt_data = gt_data[:, :480000]
    gt_data = torch.mean(gt_data, dim=0, keepdim=True)
    gt_data = gt_data.reshape(1, -1).cpu().numpy()
    gt_embed = clap_model.get_audio_embedding_from_data(x = gt_data, use_tensor=False)

    output_data, sr = torchaudio.load(output_file)
    output_data = torchaudio.functional.resample(output_data, orig_freq=sr, new_freq=48000)
    output_data = output_data[:, :480000]
    output_data = torch.mean(output_data, dim=0, keepdim=True)
    output_data = output_data.reshape(1, -1).cpu().numpy()
    output_embed = clap_model.get_audio_embedding_from_data(x = output_data, use_tensor=False)

    cosine_similarity = np.dot(gt_embed, output_embed.T) / (np.linalg.norm(gt_embed) * np.linalg.norm(output_embed))
    print(gt_files[i])
    print(f"Similarity: {cosine_similarity[0][0]}")
    avg_sim.append(cosine_similarity[0][0])

print(f"avg similarity: {sum(avg_sim)/len(avg_sim)}")