from denoiser import pretrained
from denoiser.dsp import convert_audio
import torch

denoiser_model = pretrained.dns64().cpu()

def denoise_audio(audio_samples, sample_rate = 16000):
    wav_tensor = torch.tensor(audio_samples).float().cpu()

    if sample_rate != denoiser_model.sample_rate:
        wav_tensor = convert_audio(wav_tensor, sample_rate, denoiser_model.sample_rate, denoiser_model.chin)


    with torch.no_grad():
        denoised = denoiser_model(wav_tensor.unsqueeze(0))[0]

    # Display both the original and the denoised audio

    denoised_numpy = denoised.squeeze().cpu().numpy()
    return denoised_numpy