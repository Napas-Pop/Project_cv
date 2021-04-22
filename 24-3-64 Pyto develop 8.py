import pyto_ui as ui
import pyto_ui as ui
import math
from math import *
import numpy as np
import cv2
import cv2.aruco as aruco
import pandas as pd
from PIL import Image
#import location as loc
import time


#name csv
name_csv_dist = "testใหม่1.csv"
name_csv_blow = "testblow.csv"

#define frame rate limit
frame_rate = 10

#define size of ArUco
markerLength = 0.2 #m

#for track from camera calibration
matrix_coefficients = [[394.25885619, 0.00000000e+00, 176.79297187, ],  # From calibrte camera
                       [0.00000000e+00, 394.43979784, 228.18139903, ],
                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
distortion_coefficients = [[4.26629830e-01, -3.32281005e+00, -6.00006033e-04, -3.28199014e-03, 8.39347057e+00]]  # From calibrte camera





def button_pressed(sender):
    track(matrix_coefficients, distortion_coefficients,markerLength,frame_rate)  # start to track

def button_r_pressed(sender): #reset zero
    global tvec_ini
    global tvec
    global blow
    global dist
    global data_tvecs
    global value_dis
    global value_blow
    global cX
    global cY
    global cX_ini
    global cY_ini
    data_tvecs=[]
    value_dis=[]
    value_blow = []
    
    tvec_ini=tvec
    dist =0
    blow=0
    cX_ini = cX
    cY_ini = cY
    
def print_pressed(sender):
    global data_blow 
    global data_value_dis
    global name_csv_dist
    global name_csv_blow
    
    data_value_dis.to_csv(name_csv_dist)
    data_blow.to_csv(name_csv_blow)
    
    
    
def d():
    print('Distance', data_value_dis)  # show distance between tvec


# In[ ]:


def track(matrix_coefficients, distortion_coefficients,markerLength,frame_rate):
    print('check markerLengt', markerLength)
    global cX_ini
    global cY_ini
    global cY
    global cX
    global data_blow 
    global data_value_dis
    global tvec_ini
    global tvec
    global blow
    global dist
    global data_tvecs
    global value_dis
    global value_blow
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
                    cv2.circle(frame, (cX_ini, cY_ini), 4, (0, 255, 0), -1 )
                    
            fps = 1/time_elapsed
            fps = np.round(fps, 5)
            cv2.putText(frame, str(fps),
                        (50,75), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
            
            data_value_dis= pd.DataFrame(value_dis, columns = ['real Distance'])
            data_blow = pd.DataFrame(value_blow, columns = ['avg blow'])
            
            #for gui
            image_in=frame
            image_in = Image.fromarray(image_in)
            vdo.image=image_in
            diff_text="{:.3f}".format(float(dist))
            blow_show="{:.3f}".format(float(blow))
            label1.text= 'distance= ' + diff_text +' m'
            label2.text ='deform per blows='+blow_show + ' m'
            label3.text ='deform per blows realtime ='+ str(np.round(blow,4))+ ' m'
            
    return


# In[ ]:


cap = cv2.VideoCapture(0) # Get the camera source


# In[ ]:


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

button_pt = ui.Button(title="print")
button_pt.size = (100, 50)
button_pt.center = (view.width/2+100, (view.height/2+270))
button_pt.flex = [
    ui.FLEXIBLE_TOP_MARGIN,
    ui.FLEXIBLE_BOTTOM_MARGIN,
    ui.FLEXIBLE_LEFT_MARGIN,
    ui.FLEXIBLE_RIGHT_MARGIN
]
button_pt.action = print_pressed
view.add_subview(button_pt)

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
#label4 =ui.Label()
#label4.size=(view.width,50)
#label4.text_alignment=ui.TEXT_ALIGNMENT_CENTER
#label4.flex = [ui.FLEXIBLE_WIDTH]
#label4.center = (view.width/2, (view.height/2+300))
#loc.start_updating()
#label4.text = str(loc.get_location())
#view.add_subview(label4)

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
