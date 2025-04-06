# The Whisper Project

The Whisper project is mainly concerned with equipping the base Whisper model with 2 capabilities:
- Speaker diarization
- Better performance on the Polish language

The former is to be achieved by fine-tuning the model on a dataset of speeches given at the Polish parliament. The dataset should contain pairs of meeting audio recordings and their respective stenograms. The stenograms, along with the meeting video recordings can be found at [https://www.sejm.gov.pl/sejm10.nsf/stenogramy.xsp](https://www.sejm.gov.pl/sejm10.nsf/stenogramy.xsp). The audio of each meeting can be extracted from the videos. Then, the audio is to be aligned with the stenograms ([https://arxiv.org/pdf/2306.07744](https://arxiv.org/pdf/2306.07744)). Thereafter, the audio has to be resampled to the frequency of the audio Whisper was trained on. Only then can the model be fine-tuned on the gathered dataset. Additional details, reseach and fun facts can be found on the Whisper chats on Discord.
