import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import wave
import math
import contextlib
from scipy.signal import lfilter

# Function to interpret WAV file
def interpret_wav(raw_bytes, n_frames, n_channels, sample_width, interleaved=True):
    if sample_width == 1:
        dtype = np.uint8  # unsigned char
    elif sample_width == 2:
        dtype = np.int16  # signed 2-byte short
    else:
        raise ValueError("Only supports 8 and 16 bit audio formats.")

    channels = np.frombuffer(raw_bytes, dtype=dtype)

    if interleaved:
        # channels are interleaved, i.e. sample N of channel M follows sample N of channel M-1 in raw data
        channels.shape = (n_frames, n_channels)
        channels = channels.T
    else:
        # channels are not interleaved. All samples from channel M occur before all samples from channel M-1
        channels.shape = (n_channels, n_frames)

    return channels


# Function to extract audio from WAV file
def extract_audio(fname, tStart=None, tEnd=None):
    with contextlib.closing(wave.open(fname, 'rb')) as spf:
        sampleRate = spf.getframerate()
        ampWidth = spf.getsampwidth()
        nChannels = spf.getnchannels()
        nFrames = spf.getnframes()

        if tStart is None:
            startFrame = 0
        else:
            startFrame = int(tStart * sampleRate)

        if tEnd is None:
            endFrame = nFrames
        else:
            endFrame = int(tEnd * sampleRate)

        segFrames = endFrame - startFrame

        # Extract raw audio from multi-channel WAV file
        spf.setpos(startFrame)
        sig = spf.readframes(segFrames)
        spf.close()

        channels = interpret_wav(sig, segFrames, nChannels, ampWidth, True)

        return channels, nChannels, sampleRate, ampWidth, nFrames


# Function to convert audio to mono
def convert_to_mono(channels, nChannels, outputType):
    if nChannels == 2:
        samples = np.mean(channels, axis=0).astype(outputType)  # Convert to mono
    else:
        samples = channels[0]

    return samples


# Function to plot amplitude and frequency
def plot_audio_samples(title, samples, sampleRate, tStart=None, tEnd=None):
    if tStart is None:
        tStart = 0

    if tEnd is None or tStart > tEnd:
        tEnd = len(samples) / sampleRate

    plt.figure(figsize=(20, 10))
    plt.subplot(2, 1, 1)
    plt.title(title)
    plt.plot(np.linspace(tStart, tEnd, len(samples)), samples)
    plt.ylabel('Amplitude')
    plt.xlabel('Time [sec]')

    plt.subplot(2, 1, 2)
    plt.specgram(samples, Fs=sampleRate, NFFT=1024, noverlap=192, cmap='nipy_spectral', xextent=(tStart, tEnd))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    plt.tight_layout()
    plt.savefig(title + '_amplitude_frequency.png')
    #plt.show()


# Function to save audio
def save_audio(filename, sampleRate, samples):
    wavfile.write(filename, sampleRate, samples)


# Function to apply high-pass filter
def high_pass_filter(samples, fs, fH, N, outputType):
    fH = fH / fs

    # Compute sinc filter
    h = np.sinc(2 * fH * (np.arange(N) - (N - 1) / 2.))
    # Apply window
    h *= np.hamming(N)
    # Normalize to get unity gain
    h /= np.sum(h)
    # Create a high-pass filter from the low-pass filter through spectral inversion
    h = -h
    h[int((N - 1) / 2)] += 1
    # Apply the filter to the signal
    s = np.convolve(samples, h).astype(outputType)

    return s


# Function to apply low-pass filter
def low_pass_filter(samples, fs, fL, N, outputType):
    fL = fL / fs

    # Compute sinc filter
    h = np.sinc(2 * fL * (np.arange(N) - (N - 1) / 2.))
    # Apply window
    h *= np.hamming(N)
    # Normalize to get unity gain
    h /= np.sum(h)
    # Apply the filter to the signal
    s = np.convolve(samples, h).astype(outputType)

    return s


# Function to apply band-stop filter
def band_stop_filter(samples, fs, fL, fH, NL, NH, outputType):
    fH = fH / fs
    fL = fL / fs

    # Compute a low-pass filter with cutoff frequency fL
    hlpf = np.sinc(2 * fL * (np.arange(NL) - (NL - 1) / 2.))
    hlpf *= np.blackman(NL)
    hlpf /= np.sum(hlpf)
    # Compute a high-pass filter with cutoff frequency fH
    hhpf = np.sinc(2 * fH * (np.arange(NH) - (NH - 1) / 2.))
    hhpf *= np.blackman(NH)
    hhpf /= np.sum(hhpf)
    hhpf = -hhpf
    hhpf[int((NH - 1) / 2)] += 1
    # Add both filters
    if NH >= NL:
        h = hhpf
        h[int((NH - NL) / 2): int((NH - NL) / 2 + NL)] += hlpf
    else:
        h = hlpf
        h[int((NL - NH) / 2): int((NL - NH) / 2 + NH)] += hhpf
    # Apply the filter to the signal
    s = np.convolve(samples, h).astype(outputType)

    return s


# Function to extract voice
def extract_voice(samples, outputType):
    voice = (samples[0] - samples[1]).astype(outputType)

    return voice

# Function to apply reverb
def apply_reverb(samples, fs, delay, decay):
    # Create a delayed version of the signal
    delayed_signal = np.zeros_like(samples)
    delayed_signal[delay:] = samples[:-delay]

    # Apply decay to the delayed signal
    decayed_signal = delayed_signal * decay

    # Mix the original signal with the decayed signal
    reverb_samples = samples + decayed_signal

    return reverb_samples

# Function to apply echo
def apply_echo(samples, fs, delay, decay):
    # Create a delayed version of the signal
    delayed_signal = np.zeros_like(samples)
    delayed_signal[delay:] = samples[:-delay]

    # Apply decay to the delayed signal
    decayed_signal = delayed_signal * decay

    # Mix the original signal with the decayed signal
    echo_samples = samples + decayed_signal

    return echo_samples

# Function to increase pitch
def increase_pitch(samples, fs, factor):
    # Resample the signal to increase its pitch
    new_length = int(len(samples) / factor)
    pitched_samples = np.interp(np.linspace(0, len(samples) - 1, new_length), np.arange(len(samples)), samples)

    return pitched_samples


# Define the input and output filenames
input_file = 'sound1.wav'
output_file = 'output_sound1.wav'

# Extract audio from input file
channels, nChannels, sampleRate, ampWidth, nFrames = extract_audio(input_file)

# Convert audio to mono
samples = convert_to_mono(channels, nChannels, np.int16)

# Plot original audio amplitude and frequency
plot_audio_samples("Original Sound", samples, sampleRate)

# Save the original audio
save_audio(output_file, sampleRate, samples)

# Apply high-pass filter
fH = 1000.0  # cutoff frequency
N = int(math.sqrt(0.196196 + (fH / sampleRate) ** 2) / (fH / sampleRate))
samples_filtered_high_pass = high_pass_filter(samples, sampleRate, fH, N, np.int16)

# Plot amplitude and frequency after high-pass filter
plot_audio_samples("High Pass Filtered Sound", samples_filtered_high_pass, sampleRate)

# Save the audio after high-pass filter
save_audio("high_pass_filtered_" + output_file, sampleRate, samples_filtered_high_pass)

# Apply low-pass filter
fL = 500.0  # cutoff frequency
N = int(math.sqrt(0.196196 + (fL / sampleRate) ** 2) / (fL / sampleRate))
samples_filtered_low_pass = low_pass_filter(samples, sampleRate, fL, N, np.int16)

# Plot amplitude and frequency after low-pass filter
plot_audio_samples("Low Pass Filtered Sound", samples_filtered_low_pass, sampleRate)

# Save the audio after low-pass filter
save_audio("low_pass_filtered_" + output_file, sampleRate, samples_filtered_low_pass)

# Apply band-stop filter
fL = 500.0  # low cutoff frequency
fH = 1000.0  # high cutoff frequency
NL = NH = 200  # number of taps for each filter
samples_filtered_band_stop = band_stop_filter(samples, sampleRate, fL, fH, NL, NH, np.int16)

# Plot amplitude and frequency after band-stop filter
plot_audio_samples("Band Stop Filtered Sound", samples_filtered_band_stop, sampleRate)

# Save the audio after band-stop filter
save_audio("band_stop_filtered_" + output_file, sampleRate, samples_filtered_band_stop)

# Extract voice
voice_samples = extract_voice(channels, np.int16)

# Plot amplitude and frequency of extracted voice
plot_audio_samples("Extracted Voice", voice_samples, sampleRate)

# Save the extracted voice
save_audio("extracted_voice_" + output_file, sampleRate, voice_samples)

# Define reverb parameters
reverb_delay = int(0.05 * sampleRate)  # 50 ms delay
reverb_decay = 0.5

# Define echo parameters
echo_delay = int(0.1 * sampleRate)  # 100 ms delay
echo_decay = 0.3

# Define pitch increase factor
pitch_factor = 1.5

# Apply reverb
samples_with_reverb = apply_reverb(samples, sampleRate, reverb_delay, reverb_decay)
samples_with_reverb = samples_with_reverb.astype(np.int16)  # Convert to 16-bit integer format

# Plot amplitude and frequency after reverb
plot_audio_samples("Reverb Sound", samples_with_reverb, sampleRate)

# Save the audio after reverb
save_audio("reverb_" + output_file, sampleRate, samples_with_reverb)

# Apply echo
samples_with_echo = apply_echo(samples, sampleRate, echo_delay, echo_decay)
samples_with_echo = samples_with_echo.astype(np.int16)  # Convert to 16-bit integer format

# Plot amplitude and frequency after echo
plot_audio_samples("Echo Sound", samples_with_echo, sampleRate)

# Save the audio after echo
save_audio("echo_" + output_file, sampleRate, samples_with_echo)

# Increase pitch
samples_pitched_up = increase_pitch(samples, sampleRate, pitch_factor)
samples_pitched_up = samples_pitched_up.astype(np.int16)  # Convert to 16-bit integer format

# Plot amplitude and frequency after pitch increase
plot_audio_samples("Pitch Increased Sound", samples_pitched_up, sampleRate)

# Save the audio after pitch increase
save_audio("pitch_increased_" + output_file, sampleRate, samples_pitched_up)

print("Processing completed and results saved.")