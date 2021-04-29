#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

file_name = "test"
markerLength = 0.2 #m
matrix_coefficients = [[394.25885619, 0.00000000e+00, 176.79297187, ],  # From calibrte camera
                           [0.00000000e+00, 394.43979784, 228.18139903, ],
                           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
distortion_coefficients = [[4.26629830e-01, -3.32281005e+00, -6.00006033e-04, -3.28199014e-03, 8.39347057e+00]]


# In[1]:


#part setting
def setting():
    def button_setting(sender):
        global markerLength
        global file_name
        global matrix_coefficients
        global distortion_coefficients 
        if text_field1.text != "":
            markerLength= float(text_field1.text)
        if text_field1.text == "":
            markerLength = float(0.2)
        if text_field2.text != "":
            file_name = str(text_field2.text)
        if text_field2.text == "":
            file_name = "test"
        if text_field3 != "":
            matrix_coefficients = text_field3
        if text_field3 == "":
             matrix_coefficients = [[394.25885619, 0.00000000e+00, 176.79297187, ],  # From calibrte camera
                           [0.00000000e+00, 394.43979784, 228.18139903, ],
                           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
        if text_field4 != "":
            distortion_coefficients = text_field4
        if text_field4 == "":
            distortion_coefficients = [[4.26629830e-01, -3.32281005e+00, -6.00006033e-04, -3.28199014e-03, 8.39347057e+00]]  # From calibrte camera
        view2.close()
        last_ten_blows()


    view2 = ui.View()
    view2.background_color = ui.COLOR_SYSTEM_BACKGROUND

    label5 =ui.Label()
    label5.size=(view2.width,40)
    label5.text_alignment=ui.TEXT_ALIGNMENT_CENTER
    label5.flex = [ui.FLEXIBLE_WIDTH]
    label5.text = "Setting"
    view2.add_subview(label5)

    label6 =ui.Label()
    label6.size=(view2.width,100)
    label6.text_alignment=ui.TEXT_ALIGNMENT_CENTER
    label6.flex = [ui.FLEXIBLE_WIDTH]
    label6.text = "For adjust the argument\n"
    view2.add_subview(label6)

    button_set = ui.Button(title="Save")
    button_set.size = (100, 50)
    button_set.center = (view2.width/2, (view2.height/2+300))
    button_set.flex = [
        ui.FLEXIBLE_TOP_MARGIN,
        ui.FLEXIBLE_BOTTOM_MARGIN,
        ui.FLEXIBLE_LEFT_MARGIN,
        ui.FLEXIBLE_RIGHT_MARGIN]
    button_set.action = button_setting
    view2.add_subview(button_set)

    text_field1 = ui.TextField(placeholder="What the size of marker?")
    text_field1.become_first_responder()
    text_field1.width = 350
    text_field1.center = (view2.width / 2, view2.height / 2-150)
    text_field1.flex = [
        ui.FLEXIBLE_BOTTOM_MARGIN,
        ui.FLEXIBLE_TOP_MARGIN,
        ui.FLEXIBLE_LEFT_MARGIN,
        ui.FLEXIBLE_RIGHT_MARGIN
    ]
    view2.add_subview(text_field1)

    text_field2 = ui.TextField(placeholder="What the file name to export?")
    text_field2.become_first_responder()
    text_field2.width = 350
    text_field2.center = (view2.width / 2, view2.height / 2-100)
    text_field2.flex = [
        ui.FLEXIBLE_BOTTOM_MARGIN,
        ui.FLEXIBLE_TOP_MARGIN,
        ui.FLEXIBLE_LEFT_MARGIN,
        ui.FLEXIBLE_RIGHT_MARGIN
    ]
    view2.add_subview(text_field2)
    
    text_field3 = ui.TextField(placeholder="Camera Metrix")
    text_field3.become_first_responder()
    text_field3.width = 350
    text_field3.center = (view2.width / 2, view2.height / 2-50)
    text_field3.flex = [
        ui.FLEXIBLE_BOTTOM_MARGIN,
        ui.FLEXIBLE_TOP_MARGIN,
        ui.FLEXIBLE_LEFT_MARGIN,
        ui.FLEXIBLE_RIGHT_MARGIN
    ]
    view2.add_subview(text_field3)
    
    text_field4 = ui.TextField(placeholder="Distortion coefficients")
    text_field4.become_first_responder()
    text_field4.width = 350
    text_field4.center = (view2.width / 2, view2.height / 2)
    text_field4.flex = [
        ui.FLEXIBLE_BOTTOM_MARGIN,
        ui.FLEXIBLE_TOP_MARGIN,
        ui.FLEXIBLE_LEFT_MARGIN,
        ui.FLEXIBLE_RIGHT_MARGIN
    ]
    view2.add_subview(text_field4)

    ui.show_view(view2, ui.PRESENTATION_MODE_SHEET)


# In[ ]:


#part last 10 blows
def last_ten_blows():
    global markerLength
    global file_name
    global matrix_coefficients
    global distortion_coefficients 
    #name csv
    
    #name_csv_dist = n_dist +".csv"

    #name_csv_blow = n_blow +".csv"

    name_csv_check = file_name +".csv"

    #define frame rate limit
    frame_rate = 10

    #define size of ArUco
    #markerLength = 0.2 #m

    #for track from camera calibration
    #matrix_coefficients = [[394.25885619, 0.00000000e+00, 176.79297187, ],  # From calibrte camera
     #                      [0.00000000e+00, 394.43979784, 228.18139903, ],
      #                     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
    #distortion_coefficients = [[4.26629830e-01, -3.32281005e+00, -6.00006033e-04, -3.28199014e-03, 8.39347057e+00]]  # From calibrte camera


   

    def button_pressed(sender):
        track(matrix_coefficients, distortion_coefficients,markerLength,frame_rate)  # start to track
    def button_change_data(sender):
        global markerLength
        global file_name
        global matrix_coefficients
        global distortion_coefficients 
        setting()
        
    def button_r_pressed(sender): #reset zero
        global tvec_ini
        global tvec
        global blow
        global dist
        global data_tvecs
        global value_dis
        global value_blow
        global pile_drive
        global cX
        global cY
        global cX_ini
        global cY_ini
        global check

        data_tvecs=[]
        value_dis=[]
        value_blow = []
        pile_drive = []
        check = []
        tvec_ini=tvec
        dist =0
        blow=0

        cX_ini = cX
        cY_ini = cY


    # In[ ]:


    def print_pressed(sender): #Editing data collection and handling
    #many file to check the data if success it will be only file to release
        global value_blow
        global data_blow 
        global data_value_dis
        #global data_check
        global name_csv_dist
        global name_csv_blow
        global name_csv_check

        check = pd.Series(value_blow)
        check = np.float32(check.unique())
        data_blow = pd.DataFrame(value_blow, columns = ['avg blow'])
        data_check = pd.DataFrame(check, columns = ['check'])
        #data_value_dis.to_csv(name_csv_dist)
        #data_blow.to_csv(name_csv_blow)
        data_check.to_csv(name_csv_check)



    # In[1]:


    def track(matrix_coefficients, distortion_coefficients,markerLength,frame_rate):
        print('check markerLengt', markerLength)
        global cX_ini
        global cY_ini
        global cY
        global cX
        global data_blow 
        global data_value_dis
        global data_check
        global tvec_ini
        global tvec
        global blow
        global dist
        global data_tvecs
        global value_dis
        global value_blow
        global pile_drive
        global i
        data_tvecs = []  # Prepare to add tvec
        value_dis = []  # Prepare to get result distance(scalar) unit meter
        value_blow = []
        sum_tvec=0
        tvec_ini=np.array([0,0,0])
        tvec=np.array([0,0,0])
        dist=0
        prev = 0
        blow = 0.00
        cX=0
        cY=0
        cX_ini = 300
        cY_ini =50
        check = []

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
                            value_dis.append(np.round(sum_tvec, 3))  # get the distance data(scalar)  in array value
                        #calculate move per blow
                        length = len(value_dis)

                        if length>10:
                            pile_drive = []
                            for x in range(10):
                                pile_drive.append(value_dis[length-x-1])
                            #print('pile drive before', pile_drive)  

                            if np.std(pile_drive) <= .005:
                                #print('pile drive after', pile_drive)
                                blow = float(np.average(pile_drive))
                                value_blow.append(np.round(blow,4))
                                check = pd.Series(value_blow)
                                check = check.unique()
                            value_dis = []
                            pile_drive = []
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

                fps = 1/time_elapsed
                fps = np.round(fps, 1)
                cv2.putText(frame, 'F5PS  '+str(fps),
                            (50, 75), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)

                data_value_dis= pd.DataFrame(value_dis, columns = ['real Distance'])
                #data_blow = pd.DataFrame(value_blow, columns = ['avg blow'])
                #data_check = pd.DataFrame(check, columns = ['check'])
                #for gui
                image_in=frame
                image_in = Image.fromarray(image_in)
                vdo.image=image_in
                diff_text="{:.3f}".format(float(dist))
                blow_show="{:.3f}".format(float(blow))
                label1.text= 'distance= ' + diff_text +' m'
                #label2.text ='deform per blows='+blow_show + ' m'
                #label3.text ='deform per blows realtime ='+ str(np.round(blow,4))+ ' m'

        return



    cap = cv2.VideoCapture(0) # Get the camera source
    #GUI Part
    view = ui.View()
    view.background_color = ui.COLOR_SYSTEM_BACKGROUND
    #GUI botton
    button = ui.Button(title="record")
    button.size = (100, 50)
    button.center = (view.width/3+175, (view.height/2+300))
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
    button_r.center = (view.width/3-50, (view.height/2+300))
    button_r.flex = [
        ui.FLEXIBLE_TOP_MARGIN,
        ui.FLEXIBLE_BOTTOM_MARGIN,
        ui.FLEXIBLE_LEFT_MARGIN,
        ui.FLEXIBLE_RIGHT_MARGIN
    ]
    button_r.action = button_r_pressed
    view.add_subview(button_r)

    button_pt = ui.Button(title="print")
    button_pt.size = (100, 50)
    button_pt.center = (view.width/2, (view.height/2+300))
    button_pt.flex = [
        ui.FLEXIBLE_TOP_MARGIN,
        ui.FLEXIBLE_BOTTOM_MARGIN,
        ui.FLEXIBLE_LEFT_MARGIN,
        ui.FLEXIBLE_RIGHT_MARGIN
    ]
    button_pt.action = print_pressed
    view.add_subview(button_pt)
    
    button_ch = ui.Button(title="setting")
    button_ch.size = (100, 30)
    button_ch.center = (view.width/2+160, 50)
    button_ch.flex = [
        ui.FLEXIBLE_TOP_MARGIN,
        ui.FLEXIBLE_BOTTOM_MARGIN,
        ui.FLEXIBLE_LEFT_MARGIN,
        ui.FLEXIBLE_RIGHT_MARGIN
    ]
    button_ch.action = button_change_data
    view.add_subview(button_ch)

    #show distance
    label1 =ui.Label()
    label1.size=(view.width,40)
    label1.text_alignment=ui.TEXT_ALIGNMENT_CENTER
    label1.flex = [ui.FLEXIBLE_WIDTH]
    label1.text = "Pile Drive Develop ver.10"
    view.add_subview(label1)

    #show the size of marker
    label2 =ui.Label()
    label2.size=(view.width,50)
    label2.text_alignment=ui.TEXT_ALIGNMENT_CENTER
    label2.flex = [ui.FLEXIBLE_WIDTH]
    label2.center = (view.width/2, 150)
    label2.text = "|   the size of marker  "+ str(markerLength) + "   m.   |"
    view.add_subview(label2)

    #label3 =ui.Label()
    #label3.size=(view.width,50)
    #label3.text_alignment=ui.TEXT_ALIGNMENT_CENTER
    #label3.flex = [ui.FLEXIBLE_WIDTH]
    #label3.center = (view.width/2, 60)
    #label3.text = "blow realtime"
    #view.add_subview(label3)

    #show location
    label3 =ui.Label()
    label3.size=(view.width/2+100,70)
    label3.text_alignment=ui.TEXT_ALIGNMENT_CENTER
    label3.flex = [ui.FLEXIBLE_WIDTH]
    label3.center = (view.width/2,80)
    loc.start_updating()
    label3.text = str(loc.get_location())
    view.add_subview(label3)

    #Guidance on use
    label4 =ui.Label()
    label4.size=(view.width,550)
    label4.text_alignment=ui.TEXT_ALIGNMENT_CENTER
    label4.flex = [ui.FLEXIBLE_WIDTH]
    label4.text = "Guidance on use\n"+"------------------------\n"+"1.press record to open the camera\n"+"2.press reset/start for start the program\n or reset the data" + "\n3. You can press print to release data on iClound\n "+"------------------------\n"+"note: before you record please check the size \nof marker again"
    view.add_subview(label4)

    #Capture display
    vdo =ui.ImageView()
    vdo.size = (400, 425)
    vdo.center = (view.width/2, view.height/2+30)
    vdo.flex = [
        ui.FLEXIBLE_TOP_MARGIN,
        ui.FLEXIBLE_BOTTOM_MARGIN,
        ui.FLEXIBLE_LEFT_MARGIN,
        ui.FLEXIBLE_RIGHT_MARGIN
    ]
    view.add_subview(vdo)

    ui.show_view(view, ui.PRESENTATION_MODE_SHEET)


# In[ ]:
last_ten_blows()



