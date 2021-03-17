import pyto_ui as ui
import pyto_ui as ui
import math
from math import *
import numpy as np
import cv2
import cv2.aruco as aruco
import pandas as pd
from PIL import Image
import time


matrix_coefficients = [[394.25885619, 0.00000000e+00, 176.79297187, ],  # From calibrte camera
                       [0.00000000e+00, 394.43979784, 228.18139903, ],
                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
distortion_coefficients = [[4.26629830e-01, -3.32281005e+00, -6.00006033e-04, -3.28199014e-03, 8.39347057e+00]]  # From calibrte camera
dist = 0
sum_tvec=0
tvec_ini=np.array([0,0,0])
tvec=np.array([0,0,0])
cX=0
cY=0
cX_ini = 300
cY_ini =50
makerLength = 0.2 #m
dist_show=0



def button_pressed(sender):
    track(matrix_coefficients, distortion_coefficients,makerLength)  # start to track

def button_r_pressed(sender): #reset zero
    global tvec_ini
    global tvec
    global cX
    global cY
    global cX_ini
    global cY_ini
    global dist
    
    cX_ini = cX
    cY_ini = cY
    tvec_ini=tvec
    dist = 0.000
    dist_show = 0
def stop_pressed(sender):
    global dist
    global dist_show
    dist_show = dist
    
def d():
    print('Distance', data_value_dis)  # show distance between tvec




def track(matrix_coefficients, distortion_coefficients,markerLength):
    global tvec
    global cX_ini
    global cY_ini
    global cY
    global cX
    global dist_show
    global dist
    sum_tvec=0
    dist=0
    fpsLimit = 0.01  # throttle limit
    startTime = time.time()
    print('check markerLength',markerLength)

    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        nowTime = time.time()
        
        if (int(nowTime - startTime)) > fpsLimit:
            
            # operations on the frame come here
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Change grayscale
            aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)  # Use 5x5 dictionary to find markers
            parameters = aruco.DetectorParameters_create()  # Marker detection parameters
            # lists of ids and the corners beloning to each id
            corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict,
                                                                    parameters=parameters,
                                                                    cameraMatrix=np.float32(matrix_coefficients),
                                                                    distCoeff=np.float32(distortion_coefficients))



            if np.all(ids is not None):  # If there are markers found by detector
                for i in range(0, len(ids)):  # Iterate in markers
            # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
                    rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], markerLength, np.float32(matrix_coefficients),
                                                                       np.float32(distortion_coefficients))

                    (rvec - tvec).any()  # get rid of that nasty numpy value array error
                    aruco.drawDetectedMarkers(frame, corners)  # Draw A square around the markers

                    diff = tvec - tvec_ini
                    sum_tvec = sqrt(np.sum(diff**2)) #convert vector to scalar
                    dist = np.round(sum_tvec,4) #get the distance data(scalar)  in array value
                    #distacne show by dist

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
                    cv2.circle(frame, (cX_ini, cY_ini), 4, (0, 255, 0), -1 )
                    #cv2.putText(frame, 'Initial Point'+ str(tvec_ini),
                     #           (cX_ini+15, cY_ini+15), cv2.FONT_HERSHEY_SIMPLEX,
                      #          0.5, (0, 255, 0), 2)
            image_in=frame
            image_in = Image.fromarray(image_in)
            vdo.image=image_in
            diff_text=str(dist_show)
            #label1.text= 'tvec initial' 
            #label3.text = str(tvec_ini) 
            label1.text = 'distance run=' + str(dist)+'   m'
            label3.text= 'distance stop= ' + diff_text +' m'

            startTime = time.time()  # reset time
        
    return





cap = cv2.VideoCapture(0)  # Get the camera source

#GUI Part
view = ui.View()
view.background_color = ui.COLOR_SYSTEM_BACKGROUND
#GUI botton
button = ui.Button(title="record")
button.size = (100, 50)
button.center = (view.width/2+100, (view.height/2+300))
button.flex = [
    ui.FLEXIBLE_TOP_MARGIN,
    ui.FLEXIBLE_BOTTOM_MARGIN,
    ui.FLEXIBLE_LEFT_MARGIN,
    ui.FLEXIBLE_RIGHT_MARGIN
]
button.action = button_pressed
view.add_subview(button)

button_r = ui.Button(title="reset/start")
button_r.size = (100, 50)
button_r.center = (view.width/2-100, (view.height/2+300))
button_r.flex = [
    ui.FLEXIBLE_TOP_MARGIN,
    ui.FLEXIBLE_BOTTOM_MARGIN,
    ui.FLEXIBLE_LEFT_MARGIN,
    ui.FLEXIBLE_RIGHT_MARGIN
]
button_r.action = button_r_pressed
view.add_subview(button_r)

button_st = ui.Button(title="stop")
button_st.size = (100, 50)
button_st.center = (view.width/2-100, (view.height/2+270))
button_st.flex = [
    ui.FLEXIBLE_TOP_MARGIN,
    ui.FLEXIBLE_BOTTOM_MARGIN,
    ui.FLEXIBLE_LEFT_MARGIN,
    ui.FLEXIBLE_RIGHT_MARGIN
]
button_st.action = stop_pressed
view.add_subview(button_st)

#show distance
label1 =ui.Label()
label1.size=(view.width,20)
label1.text_alignment=ui.TEXT_ALIGNMENT_CENTER
label1.flex = [ui.FLEXIBLE_WIDTH]
label1.text = "Test Pyto Develop ver 4"
view.add_subview(label1)

label2 =ui.Label()
label2.size=(view.width,90)
label2.text_alignment=ui.TEXT_ALIGNMENT_CENTER
label2.flex = [ui.FLEXIBLE_WIDTH]
label2.text = ""
view.add_subview(label2)

label3 =ui.Label()
label3.size=(view.width,55)
label3.text_alignment=ui.TEXT_ALIGNMENT_CENTER
label3.flex = [ui.FLEXIBLE_WIDTH]
label3.text = ""
view.add_subview(label3)

label4 =ui.Label()
label4.size=(view.width,125)
label4.text_alignment=ui.TEXT_ALIGNMENT_CENTER
label4.flex = [ui.FLEXIBLE_WIDTH]
label4.text = ""
view.add_subview(label4)

#Capture display
vdo =ui.ImageView()
vdo.size = (400, 470)
vdo.center = (view.width/2, view.height/2-20)
vdo.flex = [
    ui.FLEXIBLE_TOP_MARGIN,
    ui.FLEXIBLE_BOTTOM_MARGIN,
    ui.FLEXIBLE_LEFT_MARGIN,
    ui.FLEXIBLE_RIGHT_MARGIN
]
view.add_subview(vdo)

ui.show_view(view, ui.PRESENTATION_MODE_SHEET)

