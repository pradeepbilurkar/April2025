import cv2
import matplotlib.pyplot as plt
from skimage.segmentation import active_contour
from skimage.color import rgb2gray
import numpy as np
from skimage.morphology import disk
from skimage.measure import label, shannon_entropy
from skimage.feature import graycomatrix, graycoprops
from skimage.filters.rank import entropy
from skimage.filters import threshold_otsu
from skimage.measure import regionprops, regionprops_table
from skimage.segmentation import clear_border
from skimage import img_as_ubyte
import pandas as pd



image_path = 'C:/Users/Samsan/Downloads/pi1.jpg'
Im = cv2.imread(image_path)
#Im=cv2.resize(Im, (600, 600),interpolation=cv2.INTER_LINEAR)
# Im=cv2.cvtColor(Im, cv2.COLOR_BGR2GRAY)
# Im=cv2.bitwise_not(Im1)
# new_image=cv2.resize(im1, (600, 600),interpolation=cv2.INTER_LINEAR)

ROTATION_COUNT = 72
# Load the image
# Get image dimensions
h, w = Im.shape[:2]
center = (w // 2, h // 2)

for c in range(1, ROTATION_COUNT + 1):
    print(f"Processing {c}")  # Equivalent to `app.ProcessButton.Text = num2str(c)`
    # Compute rotation matrix
    try:
        Im = cv2.cvtColor(Im, cv2.COLOR_BGR2GRAY)
    except:
        1
    angle = -c * 5  # MATLAB's -c.*5 equivalent
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Rotate image (crop mode)
    IRot_original = cv2.warpAffine(Im, rotation_matrix, (w, h))
    IRot=IRot_original.copy()
    # Display rotated image (optional)
    # x_start, y_start, width, height = 900, 350, 550, 200
    x_start, y_start, width, height = 950, 380, 120, 180
    # Perform cropping
    im_roi = IRot[y_start:y_start + height, x_start:x_start + width]
    #im_roi = cv2.cvtColor(im_roi, cv2.COLOR_GRAY2RGB)
    #cv2.imshow('Final', im_roi)
    #cv2.waitKey(0)
    Ip = im_roi
    # Apply threshold

    Ip[Ip < 100] = 0
    # Apply dilation using a disk structuring element (similar to offsetstrel)
    se = disk(18)
    BW2 = cv2.dilate(Ip, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    # Compute Otsu threshold
    level = threshold_otsu(BW2)
    level = level + (0.1 * level)  # Adjust threshold like MATLAB
    # Binarize Image
    I_F = BW2 > level
    I_F = I_F.astype(np.uint8)  # Convert to uint8
    I_F = img_as_ubyte(I_F)  # Ensure correct format
    # I_F[I_F > 0] = 255
    I_F[(I_F >= 100) & (I_F <= 200)] = 0
    I_F[I_F > 210] = 0

    # Make binary (0 or 255)
    I_F = cv2.cvtColor(BW2, cv2.COLOR_BGR2GRAY) if BW2.ndim == 3 else BW2  # Convert to grayscale if needed
    # cv2.imshow('Final', I_F)
    # cv2.waitKey(0)
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((I_F), connectivity=8)
    # skimage_labels = label(labels)
    labels = label(labels)
    # Extract properties like MATLAB regionprops()
    regions = regionprops_table(labels, properties=['area', 'eccentricity', 'orientation', 'bbox', 'centroid',
                                                    'major_axis_length', 'minor_axis_length'])
    I_F_colored = cv2.cvtColor(I_F, cv2.COLOR_GRAY2BGR)
    # im_roi_coloured = cv2.cvtColor(im_roi, cv2.COLOR_GRAY2BGR)
    distances = [1]  # Pixel pair distance
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # Different angles

    # Compute GLCM for each labeled region
    glcm_features = {}

    for i in range(len(regions['bbox-0'])):
        xmin = regions['bbox-1'][i]
        ymin = regions['bbox-0'][i]
        xmax = regions['bbox-3'][i]
        ymax = regions['bbox-2'][i]

        region_image = im_roi[ymin:ymax, xmin:xmax]
        if region_image.size > 0:
            # Compute GLCM

            glcm = graycomatrix(region_image.astype(np.uint8), distances, angles, symmetric=True, normed=True)

            # Compute contrast and other texture features
            contrast = graycoprops(glcm, 'contrast')
            dissimilarity = graycoprops(glcm, 'dissimilarity')
            homogeneity = graycoprops(glcm, 'homogeneity')
            energy = graycoprops(glcm, 'energy')
            correlation = graycoprops(glcm, 'correlation')

            # Store features
            glcm_features[i] = {
                "contrast": contrast.mean(),
                "dissimilarity": dissimilarity.mean(),
                "homogeneity": homogeneity.mean(),
                "energy": energy.mean(),
                "correlation": correlation.mean(),
            }
            cv2.rectangle(im_roi, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Draw green boxes
    # for i in range(len(centroids)):
    #     cv2.circle(im_roi, (int(centroids[i][1]), int(centroids[i][0])), 3, (255, 0, 0),
    #                -1)  # Red dot for centroid
    #
    #cv2.imshow('Regions Highlighted', im_roi)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Convert to usable format
    #stats_df = pd.DataFrame(props)
    Stats = []
    for i in range(len(regions["area"])):
        stat = {
            "Area":regions['area'][i],
            "Orientation":regions['orientation'][i],
            "eccentricity": regions['eccentricity'][i],
            "Major": regions["major_axis_length"][i] ,
            "Minor": regions["minor_axis_length"][i] ,
            "BBX": regions["bbox-1"][i],
            "BBY": regions["bbox-0"][i],
            "CenX": regions["centroid-1"][i],
            "CenY": regions["centroid-0"][i],
            "BBXRange": regions["bbox-1"][i] + regions["bbox-3"][i],
            "BBYRange": regions["bbox-0"][i] + regions["bbox-2"][i],
        }
        Stats.append(stat)

    idx = [
         m for m, stat in enumerate(Stats)
         if (900 < stat["Area"] < 1000) and (-.9 < stat["Orientation"] <0) and
         (30 < stat["Major"] < 50) and (stat["Minor"] > 20) and (80<stat['BBX']<90)
            and (glcm_features[m]['contrast']<100)
     ]

    #IFinal_comp = img_as_ubyte(np.invert(BW2))  # Equivalent to `imcomplement` & `im2uint8`
    # Extract filtered stats
    stats_results = [Stats[z] for z in idx]
    #isualize_blob_properties(im_roi, 0,stats_results, 0,0, im_roi, in_app=False, ax=None)
    if len(IRot_original.shape) == 2:  # If grayscale, convert to RGB
        IRot_original = cv2.cvtColor(IRot_original, cv2.COLOR_GRAY2BGR)
        1
    if len(idx) >0:
        for k in range(len(stats_results)):
            xbar = stats_results[k]["CenX"] + x_start
            ybar = stats_results[k]["CenY"] + y_start
            a = stats_results[k]["Major"] / 2
            b = stats_results[k]["Minor"] / 2
            theta = np.radians(stats_results[k]["Orientation"])

            R = np.array([[np.cos(theta), np.sin(theta)],
                          [-np.sin(theta), np.cos(theta)]])

            phi = np.linspace(0, 2 * np.pi, 50)
            cosphi = np.cos(phi)
            sinphi = np.sin(phi)

            xy = np.vstack((a * cosphi, b * sinphi))
            xy = R @ xy

            x = (xy[0, :] + xbar).astype(int)
            y = (xy[1, :] + ybar).astype(int)

            for i in range(len(x) - 1):
                cv2.line(IRot_original, (x[i], y[i]), (x[i + 1], y[i + 1]), (0, 0, 255), 1)  # Draw ellipses on main image
        #Resized_IRot=cv2.resize(IRot_original, (900,900), interpolation=cv2.INTER_LINEAR

        cv2.imshow("Contours on Main Image", IRot_original)
        cv2.moveWindow("Contours on Main Image", 100, 100)
        cv2.namedWindow("Contours on Main Image", cv2.WINDOW_NORMAL)
        #cv2.resizeWindow("Contours on Main Image", 400, 400)  # Set initial window size
        cv2.waitKey(0)
        cv2.destroyAllWindows(),
        break
    else:
        try:
            Im= cv2.cvtColor(Im, cv2.COLOR_RGB2GRAY)
        except:
            1