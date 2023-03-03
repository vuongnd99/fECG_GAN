# Fetal ECG Extraction on Time-Frequency Domain using Image-to-image Translation GANs

This repository contains code and data for implementing a method to extract fetal electrocardiogram (ECG) signals from a maternal ECG signal on the time-frequency domain using image-to-image translation generative adversarial networks (GANs).

## 1. Background

Fetal ECG extraction from maternal ECG signals is important in clinical settings, as it can provide crucial information about fetal health. However, fetal ECG signals are typically much weaker than maternal ECG signals, making it challenging to extract them directly from the raw signal. One approach is to first convert the signal into the time-frequency domain using a spectrogram, and then use image processing techniques to extract the fetal ECG signal from the spectrogram.

## 2. Method

In this repository, we present a method for fetal ECG extraction on the time-frequency domain using image-to-image translation GANs. We train a GAN to learn the mapping between maternal ECG spectrograms and corresponding fetal ECG spectrograms. During testing, we apply the trained GAN to a maternal ECG spectrogram to obtain the corresponding fetal ECG spectrogram, and then use image processing techniques to extract the fetal ECG signal from the spectrogram.

## 3. Repository Contents

- `data/`: contains sample maternal and fetal ECG signals
- `models/`: contains the checkpoints
- `src/`: contains scripts for training and testing the GAN model
- `README.md`: this file

### 3.1. Requirements

- Python 3.x
- PyTorch
- NumPy
- Matplotlib

### 3.2. Usage

Download or clone the repository:

```python
git clone [https link to my repository]
```


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
