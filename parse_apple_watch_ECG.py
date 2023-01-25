import cv2
import numpy as np
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
from scipy.signal import find_peaks, resample, savgol_filter
import sys
import uuid
import os
import json

def remove_outliers(data):
    '''
    Removes outliers from an array of numbers.

    Parameters:
    data (list): A list of numbers.

    Returns:
    list: A list of numbers with outliers removed.
    '''
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    return [x for x in data if x >= lower_bound and x <= upper_bound]

def find_bounds(spacing, coords):
    '''
    Finds the pixel bounding boxes of the ECG.

    Parameters:
    spacing (list): A list of numbers representing the spacing of the ECG.
    coords (list): A list of coordinates of the rough center of the ECG traces.

    Returns:
    list: A list of bounding y-coordinates for the ECG traces in the image.
    '''
    bounds = []
    for coord in coords:
        lower_bound = coord
        upper_bound = coord

        while lower_bound > 0:
            if spacing[lower_bound] > 0:
                lower_bound -= 1
            else:
                break
        while upper_bound < len(spacing):
            if spacing[upper_bound] > 0:
                upper_bound += 1
            else:
                break
        bounds.append([lower_bound, upper_bound])
    return bounds

def process_pdf(fn):
    '''
    Processes a PDF file containing an ECG.

    Parameters:
    fn (str): The filename of the PDF file.

    Returns:
    numpy.ndarray: An array of ECG values.
    '''

    #extract metadata from PDF
    reader = PdfReader(fn)
    text = reader.pages[0].extract_text()
    parsed = text.split(' ')
    mm_per_s = int(parsed[parsed.index('mm/s,')-1].split('\n')[-1])
    mm_per_mV = int(parsed[parsed.index('mm/mV,')-1].split('\n')[-1])

    Name = text.split('\n')[0]
    DOB = text.split('\n')[1].split('Date of Birth:')[-1].split('(')[0].strip()
    Age = int(text.split('\n')[1].split('(Age')[-1].split(')')[0].strip())
    recording_time = ':'.join([item.strip() for item in text.split('\n')[1].split('Recorded on ')[-1].strip().split(':')])

    #get frequency from metadata
    for item in parsed:
        if "Hz" in item:
            fs = int(''.join(filter(str.isdigit, item)))

    metadata = {"frequency": fs, "Name": Name, "DOB": DOB, "Age": Age, "Recording time": recording_time}

    temp_fn = str(uuid.uuid4()) #create temporary filename for the working image file
    pages = convert_from_path(fn, 800) #extract PDF as image
    pages[0].save(f'{temp_fn}.jpg', 'JPEG') #save image as jpg for later use

    img = cv2.imread(f'{temp_fn}.jpg') #load the saved PDF as a cv2 image
    ecg_only = img * (np.repeat(np.sum((img - [35,10,195])**2, axis=-1)[:, :, np.newaxis], 3, axis=-1) < 10000) #filter out all pixels but the ECG

    spacing = np.sum(np.sum(ecg_only, axis=-1), axis=-1) #determine the spacing of the ECG traces (since it's broken into multiple lines)
    ecg_coords = find_peaks(spacing, height=np.max(spacing)/3*2, distance=10)[0] #find the coordinates of the rough center of the ECG traces

    bounds = find_bounds(spacing, ecg_coords) #find the bounding boxes for the ECG traces in the image

    bands = [] #extract the ECG bands
    for bound in bounds:
        band = ecg_only[bound[0]:bound[1]] #crop the image to the ECG in question
        band_exists = np.arange(0, band.shape[1])[np.sum(np.sum(band, axis=-1), axis=0) > 0] #find whitespace
        start = np.min(band_exists)
        end = np.max(band_exists)
        bands.append(band[:, start:end]) #only append the non-whitespace

    traces = [] #now we're going to extract the actual ECG trace values
    for band in bands:
        trace = []
        for i in range(0, band.shape[1]):
            trace.append(band.shape[0] - np.mean(np.arange(0, len(band))[np.sum(band[:, i], axis=-1) > 0]))
        traces.append(np.asarray(trace))

    for i in range(0, len(traces)-1):
        traces[i+1] += traces[i][-1] - traces[i+1][0] #we want to align the voltage values of each separate trace

    ecg = np.concatenate(traces) #concatenate the traces, and that's it!

    #now, we must figure out the spacing dimensions
    gridlines = img * (np.repeat(np.sum((img - [205,205,205])**2, axis=-1)[:, :, np.newaxis], 3, axis=-1) < 1000) #extract just the gridlines from the image
    spacing = np.sum(np.sum(gridlines, axis=-1), axis=0) #sum down vertical axis of image to detect gridlines

    peaks = find_peaks(spacing, height=np.max(spacing)/2, distance=10)[0] #find the pixel coords of the gridlines
    diffs = remove_outliers(peaks[1:]-peaks[:-1]) #remove the outliers for better accuracy
    pixels_per_sec = np.mean(diffs)/0.2 #now, compute the mean spacing and convert it to pixels/sec
    pixels_per_mV =  pixels_per_sec / mm_per_s * mm_per_mV #using pixels/sec, compute pixels/mV

    ecg /= pixels_per_mV #convert y-axis to mV
    ecg = savgol_filter(ecg, 51, 3) #filter signal to smooth out

    ecg = resample(ecg, int(fs*len(ecg)/pixels_per_sec)) #resample to recording frequency

    os.remove(f'{temp_fn}.jpg') #clean-up

    return ecg, metadata

if __name__ == "__main__":
    #Ensure correct usage
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} [filename]")
        quit()

    #Get filename from args
    fn = sys.argv[1]
    processed = process_pdf(fn)
    np.save('.'.join(fn.split('.')[:-1]) + ".npy", processed[0])
    with open('.'.join(fn.split('.')[:-1]) + "_metadata.json", 'w') as f:
        json.dump(processed[1], f)
