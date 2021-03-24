#!/usr/bin/env python
# coding: utf-8

# In[15]:


import math
from math import *
import numpy as np
import cv2
import cv2.aruco as aruco
import pandas as pd
from PIL import Image
#import location as loc
import time

#location excel
path = r"D:\2563-4\2-2563\Project\test23364.xlsx"

#define frame rate limit
frame_rate = 10

#define size of ArUco
markerLength = 0.2 #m

#for track from camera calibration
matrix_coefficients = [[394.25885619, 0.00000000e+00, 176.79297187, ],  # From calibrte camera
                       [0.00000000e+00, 394.43979784, 228.18139903, ],
                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
distortion_coefficients = [[4.26629830e-01, -3.32281005e+00, -6.00006033e-04, -3.28199014e-03, 8.39347057e+00]]  # From calibrte camera


# In[16]:


def track(matrix_coefficients, distortion_coefficients,markerLength,frame_rate):
    data_tvecs = []  # Prepare to add tvec
    value_dis = []  # Prepare to get result distance(scalar) unit meter
    value_blow = []
    sum_tvec=0
    tvec_ini=np.array([0,0,0])
    tvec=np.array([0,0,0])
    dist=0
    prev = 0
    blow = 0.00
    print('check markerLengt', markerLength)

    while cap.isOpened():
         #convert image to gray scale image
         #Capture frame-by-frame
        time_elapsed = time.time() - prev
        ret, frame = cap.read()
        
             #operations on the frame come here
        if not ret:
            continue
            
        if time_elapsed > 1./frame_rate :
            prev = time.time()
            #frame = cv2.resize(frame, (4000, 5000))
            #frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Change grayscale
            aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)  # Use 5x5 dictionary to find markers
            parameters = aruco.DetectorParameters_create()  # Marker detection parameters
            # lists of ids and the corners beloning to each id
            corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict,
                                                                    parameters=parameters,
                                                                    cameraMatrix=np.float32(matrix_coefficients),
                                                                    distCoeff=np.float32(distortion_coefficients))
            # calculate(ids,corners,matrix_coefficients,distortion_coefficients,frame)
            if np.all(ids is not None):  # If there are markers found by detector
                for i in range(0, len(ids)):  # Iterate in markers
                    # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
                    rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], markerLength,
                                                                               np.float32(matrix_coefficients),
                                                                               np.float32(distortion_coefficients))

                    (rvec - tvec).any()  # get rid of that nasty numpy value array error
                    aruco.drawDetectedMarkers(frame, corners)  # Draw A square around the markers
                    #aruco.drawAxis(frame, np.float32(matrix_coefficients), np.float32(distortion_coefficients),np.float32(rvec), np.float32(tvec), 0.01)  # Draw Axis

                    data_tvecs.append(tvec)  # get every data of tvec in array data_tvecs
                    length = len(data_tvecs)  # For easy to calculate equation
                    if length > 1:
                        diff = data_tvecs[length - 1] - tvec_ini  # Find distance between tvec(vector)
                        sum_tvec = sqrt(np.sum(diff ** 2))  # convert vector to scalar
                        dist=sum_tvec
                        value_dis.append(np.round(sum_tvec, 4))  # get the distance data(scalar)  in array value
                    #calculate move per blow
                    length =len(value_dis)
                    if length > 10:
                        pile_drive = []
                        x=0
                        for x in range(5):
                            pile_drive.append(value_dis[length-x-1])
                        if np.std(pile_drive) > .005:
                            blow=float(np.average(pile_drive))-float(blow)
                            value_blow.append(blow)
            if len(corners) > 0:
                # flatten the ArUco IDs list
                ids = ids.flatten()
                # loop over the detected ArUCo corners
                for (markerCorner, markerID) in zip(corners, ids):
                    # extract the marker corners (which are always returned in
                    # top-left, top-right, bottom-right, and bottom-left order)
                    find_center =corners.copy()
                    find_center = markerCorner.reshape((4, 2))
                    (topLeft, topRight, bottomRight, bottomLeft) = find_center
                    # convert each of the (x, y)-coordinate pairs to integers
                    topRight = (int(topRight[0]), int(topRight[1]))
                    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                    topLeft = (int(topLeft[0]), int(topLeft[1]))
                    # draw the bounding box of the ArUCo detection
                    #cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
                    #cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
                    #cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
                    #cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)
                    # compute and draw the center (x, y)-coordinates of the ArUco
                    # marker
                    cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                    cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                    cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
                    # draw the ArUco marker ID on the image
                    cv2.putText(frame, str(markerID),
                                (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2)
                    cv2.putText(frame, str(topRight),
                                (topRight[0], topRight[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2)
                    
                    #cv2.circle(frame, (cX_ini, cY_ini), 4, (0, 255, 0), -1 )
            fps = 1/time_elapsed
            fps = np.round(fps, 5)
            cv2.putText(frame, str(fps),
                        (50,75), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
            cv2.imshow('frame',frame)
            
            data_value_dis= pd.DataFrame(value_dis, columns = ['real Distance'])
            data_blow = pd.DataFrame(value_blow, columns = ['avg blow'])
            
            # Wait 3 milisecoonds for an interaction. Check the key and do the corresponding job.
            key = cv2.waitKey(3) & 0xFF
            if key == ord('q'):  # Quit
                 #show distance between tvec
                #data_value_dis.to_excel("Test1.xlsx", sheet_name='Sheet_name_1')
                ch = 'q'
                cap.release()
                cv2.destroyAllWindows()
                break
            if key == ord('p'):  # Quit
                 #show distance between tvec
                #.to_excel("Test1 23364.xlsx", sheet_name='Sheet_name_1')
                writer = pd.ExcelWriter(path, engine = 'xlsxwriter')
                data_value_dis.to_excel(writer, sheet_name = 'real')
                data_blow.to_excel(writer, sheet_name = 'avg')
                writer.save()
                writer.close()
                ch = 'p'
                
            if key == ord('r'):
                ch = 'r'
                print('Restart the program.')
                cap.release()
                cv2.destroyAllWindows()
                break

            if key == ord('s'):
                ch = 's'
                print('Start program to get the data.')
                cap.release()
                cv2.destroyAllWindows()
                break
            startTime = time.time()
            
    
    return ch


# In[17]:


device = 0 #'rtsp://192.168.1.40:8080/h264_pcm.sdp'
print('press s to start the program or q to exit the program')
print('If you want to reset the data press r')
cap = cv2.VideoCapture(device)  # Get the camera source
choice = track(matrix_coefficients, distortion_coefficients,markerLength,frame_rate) #start to track 

if choice == 's':
    while True:
        data_tvecs=[] #clear the array that get the data tvec
        value_dis = [] #clear the array that get the data distance
        value_blow = [] #clear the array that get the data average distance 
        cap = cv2.VideoCapture(device)
        print('If you want to reset the data press r')
        run_p = track(matrix_coefficients, distortion_coefficients,markerLength,frame_rate)
        if run_p == 'q':
            print('Finish the program loop 2 ')
            break
        else:
            continue
if choice == 'q':
    print('Cancle the program')

