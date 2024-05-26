import cv2
import numpy as np
import pyaudio
import wave
import os
import threading

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
        "images": [os.path.join(current_dir, 'img1.png')],
        "sounds": [os.path.join(current_dir, 'sound1.wav')],
        "sound_played": False  # Add a flag to keep track of whether the sound has been played
    },
    2: {
        "images": [os.path.join(current_dir, 'img2.png')],
        "sounds": [os.path.join(current_dir, 'sound2.wav')],
        "sound_played": False
    },
    3: {
        "images": [os.path.join(current_dir, 'img3.png')],
        "sounds": [os.path.join(current_dir, 'sound3.wav')],
        "sound_played": False
    }
}

# Audio playback setup
def play_audio(file):
    wf = wave.open(file, 'rb')
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    data = wf.readframes(1024)
    while data:
        stream.write(data)
        data = wf.readframes(1024)

    stream.stop_stream()
    stream.close()

def play_audio_thread(file):
    audio_thread = threading.Thread(target=play_audio, args=(file,))
    audio_thread.start()

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
    images = files["images"]
    sounds = files["sounds"]
    for idx, (img_path, sound_path) in enumerate(zip(images, sounds)):
        marker_image = cv2.aruco.generateImageMarker(dictionary, marker_id, 700)
        bordered_marker = np.ones((1500, 1500), dtype=np.uint8) * 255  # White background
        bordered_marker[400:1100, 400:1100] = marker_image  # Overlay marker onto white background
        marker_path = os.path.join(output_dir, f"marker_{marker_id}_{idx+1}.png")
        cv2.imwrite(marker_path, bordered_marker)
        print(f"Marker ID {marker_id}_{idx+1} saved as {marker_path}")

print("Markers generated and saved.")

def overlay_image_on_frame(background, overlay, x, y):
    """ Overlay `overlay` image on `background` image at position (x, y) """
    bg_height, bg_width = background.shape[:2]
    overlay_height, overlay_width = overlay.shape[:2]

    if x >= bg_width or y >= bg_height:
        return background

    if x + overlay_width > bg_width:
        overlay_width = bg_width - x
        overlay = overlay[:, :overlay_width]

    if y + overlay_height > bg_height:
        overlay_height = bg_height - y
        overlay = overlay[:overlay_height]

    if overlay.shape[2] == 4:
        alpha_overlay = overlay[:, :, 3] / 255.0
        alpha_background = 1.0 - alpha_overlay

        for c in range(3):
            background[y:y+overlay_height, x:x+overlay_width, c] = (
                alpha_overlay * overlay[:, :, c] +
                alpha_background * background[y:y+overlay_height, x:x+overlay_width, c]
            )
    else:
        background[y:y+overlay_height, x:x+overlay_width] = overlay

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

                    top_left = crn[0]
                    top_right = crn[1]
                    bottom_right = crn[2]
                    bottom_left = crn[3]

                    width = int(np.linalg.norm(top_right - top_left))
                    height = int(np.linalg.norm(top_right - bottom_right))

                    # Resize overlay image to fit marker size
                    resized_img = cv2.resize(img, (width, height))

                    # Perspective transform
                    dst_pts = np.array([top_left, top_right, bottom_right, bottom_left])
                    src_pts = np.array([[0, 0], [width, 0], [width, height], [0, height]])

                    matrix, _ = cv2.findHomography(src_pts, dst_pts)
                    warped_img = cv2.warpPerspective(resized_img, matrix, (frame.shape[1], frame.shape[0]))

                    frame = overlay_image_on_frame(frame, warped_img, 0, 0)

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

cap.release()
cv2.destroyAllWindows()
p.terminate()
