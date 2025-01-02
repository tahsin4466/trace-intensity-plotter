"""Imports"""
import cupy as cp
import cv2
import matplotlib.pyplot as plt
import time
from scipy.ndimage import label


"""Constants"""
#Size of batches for GPU processing
BATCH_SIZE = 100 #Adjust if needed - 100 works for the 3080 (10gb VRAM)
#Dataset file path (put in the data folder please)
DATA_PATH = "data/dataset.avi" #Change if it's not called dataset.avi!


"""Main Script"""
def segment_and_extract_intensity(data_path=DATA_PATH, batch_size=BATCH_SIZE):
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
    frames = cp.array(frames)  #Convert all frames to CuPy array for CUDA

    #Sample frame for mask computation
    sample_frame = frames[0]
    #Apply blur
    blurred = cp.array(cv2.GaussianBlur(cp.asnumpy(sample_frame), (9, 9), 2.0))
    #Apply thresholding
    _, thresh = cv2.threshold(cp.asnumpy(blurred), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cp.array(thresh)

    #Label connected components
    labeled_array, num_features = label(cp.asnumpy(thresh))
    labeled_array = cp.array(labeled_array)
    print(f"Number of ROIs detected: {num_features}")

    #Precompute ROI masks as vectors
    masks = cp.array([labeled_array == roi for roi in range(1, num_features + 1)])
    #Initialize intensity traces as a 2D array
    intensity_traces = cp.zeros((num_features, frames.shape[0]), dtype=cp.float32)

    #Batch process frames
    for i in range(0, frames.shape[0], batch_size):
        batch = frames[i:i + batch_size]
        for roi_idx, mask in enumerate(masks):
            mean_intensities = cp.mean(batch[:, mask], axis=1)
            intensity_traces[roi_idx, i:i + batch_size] = mean_intensities

    #Stop timing
    print(f"Finished processing in {time.time() - start_time:.2f} seconds")

    #Convert intensity traces to NumPy for plotting
    intensity_traces = cp.asnumpy(intensity_traces)

    #Calculate the size of each ROI (number of pixels in each labeled region)
    roi_sizes = [cp.sum(labeled_array == roi).get() for roi in range(1, num_features + 1)]
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
    plt.imshow(cp.asnumpy(labeled_array), cmap='jet')
    plt.colorbar()
    plt.title("Segmented ROIs")
    plt.show()


"""Entry"""
segment_and_extract_intensity()