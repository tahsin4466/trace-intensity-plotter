"""Imports"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from scipy.ndimage import label


"""Constants"""
# Dataset file path (put in the data folder please)
DATA_PATH = "data/dataset.avi"  #Change if it's not called dataset.avi!


"""Main Script"""
def segment_and_extract_intensity(data_path=DATA_PATH):
    #Start timer
    start_time = time.time()

    #Load the video into OpenCV
    cap = cv2.VideoCapture(data_path)
    if not cap.isOpened():
        print("Error: Couldn't open the video (is the path correct?).")
        return

    #Read and convert each frame to grayscale
    frames = []
    ret, frame = cap.read()
    while ret:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray_frame)
        ret, frame = cap.read()
    cap.release()
    frames = np.array(frames)  #Convert all frames to NumPy array for CPU processing

    #Sample frame for mask computation
    sample_frame = frames[0]
    #Apply blur
    blurred = cv2.GaussianBlur(sample_frame, (9, 9), 2.0)
    #Apply thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #Label connected components
    labeled_array, num_features = label(thresh)
    print(f"Number of ROIs detected: {num_features}")

    #Precompute ROI masks as vectors
    masks = np.array([labeled_array == roi for roi in range(1, num_features + 1)])
    #Initialize intensity traces as a 2D array
    intensity_traces = np.zeros((num_features, frames.shape[0]), dtype=np.float32)

    #Process all frames at once without batching
    for roi_idx, mask in enumerate(masks):
        intensity_traces[roi_idx] = np.mean(frames[:, mask], axis=1)

    #Stop timing
    print(f"Finished processing in {time.time() - start_time:.2f} seconds")

    #Calculate the size of each ROI (number of pixels in each labeled region)
    roi_sizes = [np.sum(labeled_array == roi) for roi in range(1, num_features + 1)]
    #Get the indices of the largest ROIs using a list comprehension
    largest_roi_indices = sorted(range(len(roi_sizes)), key=lambda x: roi_sizes[x], reverse=True)[:10]

    #Plot intensity traces for the largest 10 ROIs (or else there are way too many)
    plt.figure(figsize=(10, 6))
    for idx in largest_roi_indices:
        trace = intensity_traces[idx]
        plt.plot(trace, label=f"ROI {idx + 1}")
    plt.xlabel("Frame")
    plt.ylabel("Average Intensity")
    plt.title("Intensity Traces (Largest 10 ROIs)")
    plt.legend()
    plt.show()

    #Display segmentation result
    plt.imshow(labeled_array, cmap='jet')
    plt.colorbar()
    plt.title("Segmented ROIs")
    plt.show()


"""Entry"""
segment_and_extract_intensity()
