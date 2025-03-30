from frechet_audio_distance import FrechetAudioDistance

bg_set = "./samples/audios"
test_set = "./wav/v2m"
frechet = FrechetAudioDistance(
    model_name="clap",
    sample_rate=48000,
    submodel_name="music_audioset",
    verbose=False,
    enable_fusion=False
)

fad_score = frechet.score(
    bg_set, 
    test_set, 
    dtype="float32"
)
print(f"FAD score: {fad_score}")