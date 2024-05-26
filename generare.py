import cv2
import os

# Create an output directory for the markers
output_dir = "markers"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ArUco marker setup
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# Number of markers to generate
num_markers = 10  # Change this number to generate more or fewer markers

for marker_id in range(num_markers):
    marker_image = cv2.aruco.drawMarker(dictionary, marker_id, 700)
    marker_path = os.path.join(output_dir, f"marker_{marker_id}.png")
    cv2.imwrite(marker_path, marker_image)
    print(f"Marker ID {marker_id} saved as {marker_path}")

print("Markers generated and saved.")
