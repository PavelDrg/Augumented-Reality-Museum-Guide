import cv2
import numpy as np
import pyaudio
import wave
import os
import threading
import matplotlib.pyplot as plt
from scipy.io import wavfile
import math
import contextlib



# PRELUCRARE AUDIO ---------------------------------------------------------------

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
input_file = 'eminem.wav'
output_file = 'sample1.wav'

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






# PARTE CU AR ----------------------------------------------------------------------


# Initialize the camera and PyAudio
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

p = pyaudio.PyAudio()

# Get the current directory
current_dir = os.getcwd()

# Paths to the image and sound files
media_files = {
    1: {
        "images": [os.path.join(current_dir, 'Original Sound_amplitude_frequency.png')],
        "sounds": [os.path.join(current_dir, 'sample1.wav')],
        "sound_played": False  # Add a flag to keep track of whether the sound has been played
    },
    2: {
        "images": [os.path.join(current_dir, 'Pitch Increased Sound_amplitude_frequency.png')],
        "sounds": [os.path.join(current_dir, 'pitch_increased_sample1.wav')],
        "sound_played": False
    },
    3: {
        "images": [os.path.join(current_dir, 'Echo Sound_amplitude_frequency.png')],
        "sounds": [os.path.join(current_dir, 'echo_sample1.wav')],
        "sound_played": False
    },
    4: {
        "images": [os.path.join(current_dir, 'Reverb Sound_amplitude_frequency.png')],
        "sounds": [os.path.join(current_dir, 'reverb_sample1.wav')],
        "sound_played": False  # Add a flag to keep track of whether the sound has been played
    },
    5: {
        "images": [os.path.join(current_dir, 'Extracted Voice_amplitude_frequency.png')],
        "sounds": [os.path.join(current_dir, 'extracted_voice_sample1.wav')],
        "sound_played": False
    },
    6: {
        "images": [os.path.join(current_dir, 'High Pass Filtered Sound_amplitude_frequency.png')],
        "sounds": [os.path.join(current_dir, 'high_pass_filtered_sample1.wav')],
        "sound_played": False
    },
    7: {
        "images": [os.path.join(current_dir, 'Low Pass Filtered Sound_amplitude_frequency.png')],
        "sounds": [os.path.join(current_dir, 'low_pass_filtered_sample1.wav')],
        "sound_played": False  # Add a flag to keep track of whether the sound has been played
    },
    8: {
        "images": [os.path.join(current_dir, 'Band Stop Filtered Sound_amplitude_frequency.png')],
        "sounds": [os.path.join(current_dir, 'band_stop_filtered_sample1.wav')],
        "sound_played": False
    }
}

# Global variable to keep track of the audio stream and thread
current_audio_thread = None
stop_audio_flag = False

# Audio playback setup
def play_audio(file):
    global stop_audio_flag
    wf = wave.open(file, 'rb')
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    data = wf.readframes(1024)
    while data and not stop_audio_flag:
        stream.write(data)
        data = wf.readframes(1024)

    stream.stop_stream()
    stream.close()

def play_audio_thread(file):
    global stop_audio_flag, current_audio_thread
    stop_audio_flag = False
    current_audio_thread = threading.Thread(target=play_audio, args=(file,))
    current_audio_thread.start()

def stop_audio():
    global stop_audio_flag, current_audio_thread
    stop_audio_flag = True
    if current_audio_thread and current_audio_thread.is_alive():
        current_audio_thread.join()
    stop_audio_flag = False

# ArUco marker setup
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()

# Create an output directory for the markers
output_dir = os.path.join(current_dir, "markers")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Number of markers to generate
num_markers = len(media_files)

for marker_id, files in media_files.items():
    marker_image = cv2.aruco.generateImageMarker(dictionary, marker_id, 700)
    bordered_marker = np.ones((1500, 1500), dtype=np.uint8) * 255  # White background
    bordered_marker[400:1100, 400:1100] = marker_image  # Overlay marker onto white background
    marker_path = os.path.join(output_dir, f"marker_{marker_id}.png")
    cv2.imwrite(marker_path, bordered_marker)
    print(f"Marker ID {marker_id} saved as {marker_path}")

print("Markers generated and saved.")

def overlay_image_on_frame(background, overlay, dst_pts):
    """ Overlay `overlay` image on `background` image using the provided destination points """
    overlay_height, overlay_width = overlay.shape[:2]
    src_pts = np.array([[0, 0], [overlay_width, 0], [overlay_width, overlay_height], [0, overlay_height]])

    # Perspective transform
    matrix, _ = cv2.findHomography(src_pts, dst_pts)
    warped_overlay = cv2.warpPerspective(overlay, matrix, (background.shape[1], background.shape[0]))

    if overlay.shape[2] == 4:  # If overlay has an alpha channel
        alpha_mask = warped_overlay[:, :, 3] / 255.0
        for c in range(3):
            background[:, :, c] = background[:, :, c] * (1 - alpha_mask) + warped_overlay[:, :, c] * alpha_mask
    else:
        mask = warped_overlay.sum(axis=2) > 0  # Binary mask
        background[mask] = warped_overlay[mask]

    return background

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

    if ids is not None:
        print(f"Detected IDs: {ids}")
        for i, corner in enumerate(corners):
            crn = corner.reshape((4, 2))
            marker_id = ids[i][0]

            if marker_id in media_files:
                marker_id_info = media_files[marker_id]
                images = marker_id_info["images"]
                sounds = marker_id_info["sounds"]
                sound_played = marker_id_info["sound_played"]

                for idx, img_path in enumerate(images):
                    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                    if img is None:
                        print(f"Error loading image at {img_path}")
                        continue

                    top_left, top_right, bottom_right, bottom_left = crn

                    width = int(np.linalg.norm(top_right - top_left))
                    height = int(np.linalg.norm(top_right - bottom_right))

                    # Resize overlay image to fit marker size
                    resized_img = cv2.resize(img, (width, height))

                    # Overlay image on frame
                    frame = overlay_image_on_frame(frame, resized_img, crn)

                    if not sound_played:
                        print(f"Playing sound from {sounds[idx]}")
                        play_audio_thread(sounds[idx])
                        media_files[marker_id]["sound_played"] = True

    cv2.imshow('AR Audio Guide', frame)

    # Check for keyboard events
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('r'):
        # Reset sound_played flag when 'r' key is pressed
        for marker_info in media_files.values():
            marker_info["sound_played"] = False
    elif key & 0xFF == ord('s'):
        # Stop the currently playing sound when 's' key is pressed
        stop_audio()

cap.release()
cv2.destroyAllWindows()
p.terminate()
