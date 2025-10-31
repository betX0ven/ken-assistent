import torch
import os

def text_to_speech(text):
    device = torch.device('cpu')
    torch.set_num_threads(4)
    local_file = 'imperialwoolsilero-model-v3-ru/model.pt'

    if not os.path.isfile(local_file):
        torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/v4_ru.pt',
                                    local_file)  

    model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
    model.to(device)

    example_text = 'В недрах тундры выдры в г+етрах т+ырят в вёдра ядра кедров.'
    sample_rate = 48000
    speaker='aidar'

    audio_paths = model.save_wav(text=text,
                                speaker=speaker,
                                sample_rate=sample_rate)

text_to_speech('выполнил, сэр')