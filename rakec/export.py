from audiocraft.utils import export
from audiocraft import train
xp = train.main.get_xp_from_sig('68ae3c7c')
export.export_lm(
    xp.folder / 'checkpoint_50.th',
    './checkpoints/clapemb(spotify-small-new)/state_dict.bin'
)
# You also need to bundle the EnCodec model you used !!
# Case 1) you trained your own
# xp_encodec = train.main.get_xp_from_sig('SIG_OF_ENCODEC')
# export.export_encodec(xp_encodec.folder / 'checkpoint.th', '/checkpoints/my_audio_lm/compression_state_dict.bin')
# Case 2) you used a pretrained model. Give the name you used without the //pretrained/ prefix.
# This will actually not dump the actual model, simply a pointer to the right model to download.
export.export_pretrained_compression_model(
    'facebook/encodec_32khz',
    './checkpoints/clapemb(spotify-small-new)/compression_state_dict.bin'
)