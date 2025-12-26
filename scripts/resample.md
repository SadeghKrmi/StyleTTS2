

convert the audio .wav from 2050Hz to 2400Hz
```bash
sudo apt update
sudo apt install ffmpeg parallel -y
```

```bash
# Create the output directory
ls /mnt/d/projects/StyleTTS2-Persian-TTS/audiosets/wavs/ | wc -l
mkdir -p /mnt/d/projects/StyleTTS2-Persian-TTS/audiosets/resampled

# Run the batch conversion
find /mnt/d/projects/StyleTTS2-Persian-TTS/audiosets/wavs/ -name "*.wav" | \
parallel ffmpeg -i {} -ar 24000 -c:a pcm_s16le /mnt/d/projects/StyleTTS2-Persian-TTS/audiosets/resampled/{/}
```