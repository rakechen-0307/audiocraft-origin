import os
import torch
import torch.nn.functional as F
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

device = "cuda" if torch.cuda.is_available() else "cpu"
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

sampling_period = 1   # 1 second
image_dir = "./frames"
audio_dir = "./wav/ours"
dirs = sorted(os.listdir(image_dir))
avg_sim = []
for i in range(len(dirs)):
    file_idx = dirs[i].split('.')[0]
    files = sorted(os.listdir(os.path.join(image_dir, dirs[i])))
    sim = []
    for j in range(len(files)):
        image_file = os.path.join(image_dir, dirs[i], files[j])
        audio_file = os.path.join(audio_dir, f"{file_idx}.wav")
    
        inputs = {
            ModalityType.VISION: data.load_and_transform_vision_data([image_file], device),
            ModalityType.AUDIO: data.load_and_transform_audio_data([audio_file], device),
        }

        with torch.no_grad():
            embeddings = model(inputs)

        vision_embedding = embeddings[ModalityType.VISION]
        audio_embedding = embeddings[ModalityType.AUDIO]

        cosine_sim = F.cosine_similarity(vision_embedding, audio_embedding, dim=-1)
        sim.append(cosine_sim.item())
    
    print(dirs[i])
    print(f"cosine similarity: {sum(sim)/len(sim)}")
    avg_sim.append(sum(sim)/len(sim))

print(f"avg similarity: {sum(avg_sim)/len(avg_sim)}")