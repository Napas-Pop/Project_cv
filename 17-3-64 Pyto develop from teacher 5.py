import pyto_ui as ui
import pyto_ui as ui
import math
from math import *
import numpy as np
import cv2
import cv2.aruco as aruco
import pandas as pd
from PIL import Image
import location as loc
import time

markerLength = 0.2 #m
matrix_coefficients = [[1.25019753e+03, 0.00000000e+00, 8.31027626e+02, ],  # From calibrte camera
                       [0.00000000e+00, 1.24629197e+03, 4.93761478e+02, ],
                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
distortion_coefficients = [[0.22126322, -0.38411381, -0.01361529, 0.03204828, -0.06211731]]  # From calibrte camera
data_tvecs = []  # Prepare to add tvec
value_dis = []  # Prepare to get result distance(scalar) unit meter
sum_tvec=0
tvec_ini=np.array([0,0,0])
tvec=np.array([0,0,0])
dist_show = 0.00
def button_pressed(sender):
    track(matrix_coefficients, distortion_coefficients,markerLength)  # start to track

def button_r_pressed(sender): #reset zero
    global tvec_ini
    global tvec
    global blow
    global dist
    global data_tvecs
    global value_dis
    data_tvecs = []  
    value_dis = [] 
    tvec_ini=tvec
    dist =0
    blow=0
    
def stop_pressed(sender):
    global blow
    global dist_show
    dist_show = blow

    
def d():
    print('Distance', data_value_dis)  # show distance between tvec


def track(matrix_coefficients, distortion_coefficients,markerLength):
    global tvec
    global blow
    global dist_show
    global data_tvecs
    global value_dis
    sum_tvec=0
    blow=0.000
    dist=0
    fpsLimit = 0.01  # throttle limit
    startTime = time.time()
    print('check markerLengtg', markerLength)

    while cap.isOpened():
         #convert image to gray scale image
         #Capture frame-by-frame
        ret, frame = cap.read()
        
             #operations on the frame come here
        if not ret:
            continue
        nowTime = time.time()

        if (int(nowTime - startTime)) > fpsLimit:
             # do other cv2 stuff....

            #frame = cv2.resize(frame, (4000, 5000))
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
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




            # Display the resulting frame
            image_in=frame
            image_in = Image.fromarray(image_in)
            vdo.image=image_in
            diff_text="{:.3f}".format(float(dist))
            blow_show="{:.3f}".format(float(dist_show))
            label1.text= 'distance= ' + diff_text +' m'
            label2.text ='deform per blows='+blow_show + ' m'
            label3.text ='deform per blows realtime ='+ str(np.round(blow,4))+ ' m'

            #data_value_dis = pd.DataFrame(value_dis, columns=['Distance'])
            startTime = time.time()  # reset time

        
    return

cap = cv2.VideoCapture(2)  # Get the camera source
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m','j','p','g'))
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 10)

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
label1.size=(view.width,30)
label1.text_alignment=ui.TEXT_ALIGNMENT_CENTER
label1.flex = [ui.FLEXIBLE_WIDTH]
label1.text = "distance"
view.add_subview(label1)

#show defrom per blows
label2 =ui.Label()
label2.size=(view.width,50)
label2.text_alignment=ui.TEXT_ALIGNMENT_CENTER
label2.flex = [ui.FLEXIBLE_WIDTH]
label2.center = (view.width/2, 40)
label2.text = "defomation per blow"
view.add_subview(label2)

label3 =ui.Label()
label3.size=(view.width,50)
label3.text_alignment=ui.TEXT_ALIGNMENT_CENTER
label3.flex = [ui.FLEXIBLE_WIDTH]
label3.center = (view.width/2, 60)
label3.text = "blow realtime"
view.add_subview(label3)

#show location
#label3 =ui.Label()
#label3.size=(view.width,50)
#label3.text_alignment=ui.TEXT_ALIGNMENT_CENTER
#label3.flex = [ui.FLEXIBLE_WIDTH]
#label3.center = (view.width/2, (view.height/2+300))
#loc.start_updating()
#label3.text = str(loc.get_location())
#view.add_subview(label3)

#Capture display
vdo =ui.ImageView()
vdo.size = (400, 425)
vdo.center = (view.width/2, view.height/2)
vdo.flex = [
    ui.FLEXIBLE_TOP_MARGIN,
    ui.FLEXIBLE_BOTTOM_MARGIN,
    ui.FLEXIBLE_LEFT_MARGIN,
    ui.FLEXIBLE_RIGHT_MARGIN
]
view.add_subview(vdo)

ui.show_view(view, ui.PRESENTATION_MODE_SHEET)
'''while cap.isOpened():
#convert image to gray scale image
#Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        continue
    #frame = cv2.resize(frame, (4000, 5000))
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
 # Display the resulting frame
    image_in=frame
    image_in = Image.fromarray(image_in)
    vdo.image=image_in
'''


