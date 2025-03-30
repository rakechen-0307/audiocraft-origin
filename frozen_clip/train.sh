python3 train.py \
    --tsn_sampling \
    --data_root ./clipclap/ \
    --list_txt ./list.txt \
    --backbone_path ./checkpoint/ViT-L-14.pt \
    --mean 0.48145466 0.4578275 0.40821073 \
    --std 0.26862954 0.26130258 0.27577711 \
    --resume_path ./checkpoint/k400_vitl14_16f_dec4x1024.pth