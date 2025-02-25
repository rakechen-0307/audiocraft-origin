from audiocraft.utils import export
from audiocraft import train
xp = train.main.get_xp_from_sig('8df2c1e9')
export.export_lm(
    xp.folder / 'checkpoint_90.th',
    './checkpoints/clapemb(ytcharts-small)/state_dict.bin'
)
# You also need to bundle the EnCodec model you used !!
# Case 1) you trained your own
# xp_encodec = train.main.get_xp_from_sig('SIG_OF_ENCODEC')
# export.export_encodec(xp_encodec.folder / 'checkpoint.th', '/checkpoints/my_audio_lm/compression_state_dict.bin')
# Case 2) you used a pretrained model. Give the name you used without the //pretrained/ prefix.
# This will actually not dump the actual model, simply a pointer to the right model to download.
export.export_pretrained_compression_model(
    'facebook/encodec_32khz',
    './checkpoints/clapemb(ytcharts-small)/compression_state_dict.bin'
)