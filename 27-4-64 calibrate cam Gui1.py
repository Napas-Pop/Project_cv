#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Part calibrate
import cv2
import glob
import numpy as np
import pyto_ui as ui
from PIL import Image
import photos
import pyto_ui as ui
import pandas as pd

device=0
filename = 'calcam' 

#for calibrate camera
height = 6 #chase board
width = 8 #chase board
square_size = 0.03 #meter
i=0
frame =[]


def button_calsend(sender):
    sender.superview.close()
    

def button_pressed(sender):
    cap_image(filename)
    
def button_c_pressed(sender):
    global i
    global frame
    global filename
    i += 1
    showpic = cv2.imwrite(filename+str(i)+'.png', frame)
    photos.save_image(showpic)
    

def button_r_pressed(sender):
    global i
    i=0




def cap_image(filename):
    global i
    global frame
     
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        #Display 
        image_in = frame
        image_in =Image.fromarray(image_in)
        vdo.image=image_in
        numpic_text = str(i)
        label1.text = 'Get the picture  '+ numpic_text +'/50'
        label2.text = ''
    return     

cap = cv2.VideoCapture(0) 

#GUI Part
view = ui.View()
view.background_color = ui.COLOR_SYSTEM_BACKGROUND
#GUI botton
button = ui.Button(title="start")
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

button_c = ui.Button(title="capture")
button_c.size = (100, 50)
button_c.center = (view.width/2, (view.height/2+300))
button_c.flex = [
    ui.FLEXIBLE_TOP_MARGIN,
    ui.FLEXIBLE_BOTTOM_MARGIN,
    ui.FLEXIBLE_LEFT_MARGIN,
    ui.FLEXIBLE_RIGHT_MARGIN
]
button_c.action = button_c_pressed
view.add_subview(button_c)

button_cal = ui.Button(title="calibration")
button_cal.size = (100, 50)
button_cal.center = (view.width/2-100, (view.height/2+300))
button_cal.flex = [
    ui.FLEXIBLE_TOP_MARGIN,
    ui.FLEXIBLE_BOTTOM_MARGIN,
    ui.FLEXIBLE_LEFT_MARGIN,
    ui.FLEXIBLE_RIGHT_MARGIN
]
button_cal.action = button_calsend
view.add_subview(button_cal)



#show distance
label1 =ui.Label()
label1.size=(view.width,50)
label1.text_alignment=ui.TEXT_ALIGNMENT_CENTER
label1.flex = [ui.FLEXIBLE_WIDTH]
label1.text = "Calibrate camera ver.1"
view.add_subview(label1)

label2 =ui.Label()
label2.size=(view.width,350)
label2.text_alignment=ui.TEXT_ALIGNMENT_CENTER
label2.flex = [ui.FLEXIBLE_WIDTH]
label2.text = "\n\n\nGuidance on use\n" + "-------------------------\n" + "1. press start for open the camera\n" + "\n2. press capture to get the photo, don't forget to check in pyto program seetting for connet with your  iclound\n" + "\n3. you can know the number of photo by the sentence above the program\n" + "\n4. if the number of photo is enough to use to calibrate press calibration\n"+ "-------------------------\n" + "note : For this program is for checker boards A4-30 mm squares - 8x6 verticies,9x7 square"
view.add_subview(label2)



#Capture display
vdo =ui.ImageView()
vdo.size = (400, 500)
vdo.center = (view.width/2, view.height/2)
vdo.flex = [
    ui.FLEXIBLE_TOP_MARGIN,
    ui.FLEXIBLE_BOTTOM_MARGIN,
    ui.FLEXIBLE_LEFT_MARGIN,
    ui.FLEXIBLE_RIGHT_MARGIN
]
view.add_subview(vdo)

ui.show_view(view, ui.PRESENTATION_MODE_SHEET)


# In[ ]:


#for calibrate camera
height = 6 #chase board
width = 8 #chase board
square_size = 0.03 #meter
mstx=[]
dist =[]
rvec = []
tvecs = []

def button_export(sender):
    global mstx, dist, rvecs, tvecs
    export_data = {'name': ['mstx', 'dist'], 'data': [mstx, dist]}
    df = pd.DataFrame(export_data)
    df.to_csv("calibrate camera.csv")
    

def cam_cal(sender):
    calibrate_cam(height, width, square_size)
    
def calibrate_cam(height, width, square_size): #calibrate camera by picture from function cap_image
    global mstx, dist, rvecs, tvecs
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size

        # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.


    images = glob.glob('calcam*.png') #if you adjust path name please adjust variable filename too, it is the same name.
    

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

            # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

                # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)

    ret, mstx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print('mstx', mstx)
    print('dist', dist)
    label3.text = 'the number of photo can read'+ '  '+str(len(images))
    label4.text = 'Camera Metrix'
    label5.text = str(mstx)
    label6.text = 'dist=' + str(dist)



# In[ ]:


#GUI Part
view1 = ui.View()
view1.background_color = ui.COLOR_SYSTEM_BACKGROUND

#GUI botton
button_s= ui.Button(title="start")
button_s.size = (100, 50)
button_s.center = (view.width/2+100, (view.height/2+300))
button_s.flex = [
    ui.FLEXIBLE_TOP_MARGIN,
    ui.FLEXIBLE_BOTTOM_MARGIN,
    ui.FLEXIBLE_LEFT_MARGIN,
    ui.FLEXIBLE_RIGHT_MARGIN
]
button_s.action = cam_cal
view1.add_subview(button_s)

button_e = ui.Button(title="Export data")
button_e.size = (100, 50)
button_e.center = (view.width/2-100, (view.height/2+300))
button_e.flex = [
    ui.FLEXIBLE_TOP_MARGIN,
    ui.FLEXIBLE_BOTTOM_MARGIN,
    ui.FLEXIBLE_LEFT_MARGIN,
    ui.FLEXIBLE_RIGHT_MARGIN
]
button_e.action = button_export
view1.add_subview(button_e)



#show distance
label3 =ui.Label()
label3.size=(view.width,100)
label3.text_alignment=ui.TEXT_ALIGNMENT_CENTER
label3.flex = [ui.FLEXIBLE_WIDTH]
label3.text = "Calibrate calculate camera ver.1"
view1.add_subview(label3)

label4 =ui.Label()
label4.size=(view.width,300)
label4.text_alignment=ui.TEXT_ALIGNMENT_CENTER
label4.flex = [ui.FLEXIBLE_WIDTH]
label4.text = "\n\n\nGuidance on use\n" + "-------------------------\n" + "1.please press start to start calibration\n" + "\n2. please wait for the processing\n"+"\n3. the resault will show you can get the data\n if you want by press 'Export data' it will show in your icloud, the type of file is csv, it can open in excel.\n" + "-------------------------\n" 
view1.add_subview(label4)

label5 =ui.Label()
label5.size=(view.width,400)
label5.text_alignment=ui.TEXT_ALIGNMENT_CENTER
label5.flex = [ui.FLEXIBLE_WIDTH]
label5.text = ""
view1.add_subview(label5)

label6 =ui.Label()
label6.size=(view.width,600)
label6.text_alignment=ui.TEXT_ALIGNMENT_CENTER
label6.flex = [ui.FLEXIBLE_WIDTH]
label6.text = ""
view1.add_subview(label6)


ui.show_view(view1, ui.PRESENTATION_MODE_SHEET)

