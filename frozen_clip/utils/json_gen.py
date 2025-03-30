import os
import json
from tqdm import tqdm
from mutagen.mp3 import MP3

json_template = {"key": "", "artist": "", "sample_rate": 44100, "file_extension": "mp3", "description": "", "keywords": "", "duration": 0.0, "bpm": "", "genre": "", "title": "", "name": "", "instrument": "Mix", "moods": []}

dir = "../clip_spotify-small/test"
files = sorted(os.listdir(dir))
for i in tqdm(range(len(files))):
  if (files[i].split('.')[-1] == 'mp3'):
    filename = files[i].split(".")[0]
    audio = MP3(os.path.join(dir, files[i]))
    json_template['duration'] = audio.info.length
    json_template['name'] = filename
    jsonfile = os.path.join(dir, f"{filename}.json")
    json_object = json.dumps(json_template)
    with open(jsonfile, "w") as outfile:
      outfile.write(json_object)