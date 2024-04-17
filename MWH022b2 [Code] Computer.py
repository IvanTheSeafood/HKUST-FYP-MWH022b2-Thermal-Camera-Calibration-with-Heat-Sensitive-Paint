import socket
import numpy as np
import cv2
import pickle
import struct
import matplotlib.pyplot as plt
import time
from collections import deque
from decimal import *
import keyboard

mlx_shape = (24,32)
global base_temp
global top_temp

# temperature delta
delta = 0.0

# color persistence
persist = [0, 0, 0]



def colour_encode(temp):
  step = 1 / (top_temp - base_temp)
  colour = (temp - base_temp) * step
  return colour


def recv_msg(sock):
    # Read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # Read the message data
    return recvall(sock, msglen)

def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data





def colour_change(img):

    color_points = [img[225][75][2], img[75][226][1], img[225][225][2]]

    print('top right', img[225][75], 'bot left', img[75][225], 'bot right', img[225][225])
    
    output = [0, 0, 0]

    if color_points[0] >= 70:
        output[0] = 1
    
    if color_points[1] >= 100:
        output[1] = 1

    if color_points[2] >= 50:
        output[2] = 1


    # print(output)

    return output
    



def temp_extract(temp_frame, delta, region, preset):

    print("now:", region)
    print("preset:", preset)
    global persist
    print("persist", persist)

    if region[2] == 1:
        return delta

    elif region[0] != preset[0]:
        return 38.0 - temp_frame[225, 75]

    elif region[1] != preset[1]:
        return 35.0 - temp_frame[75, 225]

    else:
        return delta


def sizecheck(size_list):
    if len(size_list) < 10:
        return []
    
    avg_size = np.average(size_list)
    outliers = []
    count = 0

    for i in size_list:
        if i < 0.8*avg_size or i > 1.2*avg_size:
            outliers.append(count)
        count += 1
    
    return outliers



def caliextract(cali_arr, delta, preset):
    output = colour_change(cali_arr[0])

    delta = temp_extract(cali_arr[1], delta, output, preset)

    

    return delta, preset



    





    
    



## 70 cm

ip = "" # IP of Raspberry Pi

# start server
serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serv.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8192)
serv.bind((ip, 8080))
serv.listen()
print("SERVER: started")

# while True:
    # establish connection
conn, addr = serv.accept()

print("SERVER: connection to Client established")
i = 0
receipt = []

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print('fail')
    exit()


counter = 50
destroy = False
exist = False
caliQ = deque([], 50)

fps_counter = deque([], 10)


sqx,sqy,sqw,sqh = 0, 0, 0, 0



retry = True

while True:
    start = time.time()
    data_array = np.empty(768)
    # receive data and print
    # data = []
    # while True:
    #     packet = conn.recv(2048)
    #     if not packet: break
    #     data.append(packet)
                
    # data = pickle.loads(data)
    data = recv_msg(conn)
    data = pickle.loads(data)
        
    print('received')
    receipt.append(data)
    
    archive = data.copy()
    # print(archive)
    data_array = data
    base_temp, top_temp = data_array.min(), data_array.max()
    for i in range(len(data_array)):
        data_array[i] = colour_encode(data_array[i])

    temp_frame = (np.reshape(data_array, mlx_shape))
    # temp_frame = np.rot90(temp_frame, 1, (0,1))
    temp_frame = np.flipud(temp_frame)
    # temp_frame = np.fliplr(temp_frame)

    # print(np.shape(temp_frame))
    
    temp_frame = cv2.resize(temp_frame, (640, 480))
    temp_frame = temp_frame[0:480, 70:680]
    cv2.imshow("temp_frame", temp_frame)
    

    ret, frame = cap.read()
    if not ret:
        print('shit')
        break
    cut_frame = frame
    ori_frame = frame
    # cut_frame = np.fliplr(frame)
    cut_frame = cut_frame[0:480, 0:640-75]
    ori_frame = ori_frame[0:480, 0:640-75]
    # cv2.imshow('frame', cut_frame)

    face_frame = cut_frame.copy()
    infr_frame = temp_frame.copy()

    gray = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)
    retrycount = 0
    # cannot = False

    # start
    while retry:

        found = False
        # area detection
        sqx,sqy,sqw,sqh = 0,0,0,0
        squaresize=[100,100,100,100]
        edges = cv2.Canny(image=gray, threshold1=100, threshold2=200)
        _, threshold = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

            if len(approx)>=4 and len(approx)<=8:
                x, y, w, h = cv2.boundingRect(contour)
                midx=int(x+w/2)
                midy=int(y+h/2)
                ratio= float(w)/h
                
                if ratio <1.1 and ratio >0.8 and x>squaresize[0] and y>squaresize[1] and (x+w)<(squaresize[0]+squaresize[2]) and (y+h)<(squaresize[1]+squaresize[3]):
                    # cv2.circle(face_frame, (int(midx),int(midy)), radius=5, color=(0, 0, 255), thickness=-1)
                    # cv2.circle(infr_frame, (int(midx),int(midy)), radius=5, color=(0, 0, 255), thickness=-1)

                    if w>sqw:
                        sqx,sqy,sqw,sqh=x,y,w,h #parameters of targeted area

                        if sqw == 0 or sqh == 0:
                            if retrycount < 5:
                                retry = True
                                retrycount += 1
                                continue
                            else:
                                retry = False
                                retrycount = 0
                                print('cannnot find area, press E to retry')
                                # cannot = True
                                break
                        found = True
        
        retry = False

        if not found:
            retrycount = 0
            print('cannnot find area, press E to retry')
        # cannot = False
        retry = False





    archive_frame = (np.reshape(archive, mlx_shape))
    archive_frame = np.flipud(archive_frame)
    archive_scale = cv2.resize(archive_frame, (480, 640))
    archive_scale = archive_scale[0:480, 70:680]

    # print('Calibration temp:', archive_scale[sqx+2*sqw][sqy+2*sqh], '*C')
    # cv2.circle(face_frame, (sqx+2*sqw, sqy+2*sqh), radius=3, color=(255, 0, 255), thickness=-1)
    # cv2.circle(infr_frame, (sqx+2*sqw, sqy+2*sqh), radius=3, color=(0, 255, 0), thickness=-1)

    # if not cannot:

    # cali_frame = face_frame[sqx:sqx+2*sqw, sqy:sqy+2*sqh]

    if found:
    
        cali_frame = face_frame[sqy:sqy+2*sqh, sqx:sqx+2*sqw]

        try:
            cali_frame = cv2.resize(cali_frame, (300, 300))
            

        except cv2.error as e:
            retry = True
            continue
        
        frame_size = (2*sqw, 2*sqh)


        # cali_infra = archive_scale[sqx:sqx+2*sqw-40, sqy+20:sqy+2*sqh+20]
        cali_infra = archive_scale[sqy:sqy+2*sqh, sqx:sqx+2*sqw]

        cali_infra = cv2.resize(cali_infra, (300, 300))


        # cv2.imshow('GrayCali', cv2.cvtColor(cali_frame, cv2.COLOR_BGR2GRAY))

        cali_arr = [cali_frame, cali_infra, frame_size]
        # caliQ.append(cali_arr)

        counter = 0
        exist = True

        delta, persist = caliextract(cali_arr, delta, persist)

        cv2.circle(cali_frame, (225,75), 3, (255,0,0), -1)
        cv2.circle(cali_frame, (75,225), 3, (0,255,0), -1)
        cv2.circle(cali_frame, (225,225), 3, (0,0,255), -1)

        cv2.imshow('Cali', cali_frame)
        

        cv2.rectangle(face_frame, (sqx,sqy), (sqx+2*sqw, sqy+2*sqh), (255, 0, 255), 2)
        cv2.rectangle(infr_frame, (sqx-40,sqy+20), (sqx+2*sqw-40, sqy+2*sqh+20), (255, 0, 255), 2)
        

        palettedimen=str(sqw)+'x'+str(sqh)

    # if sqx!=0:
    #     palettedis=str(int((31/(sqw+sqh)*2*25)))+'cm'   # (control/measure*found distance)cm
    #     cv2.putText(img,palettedis,(sqx,sqy+10),3,1,(255,0,255),1)
                
    cv2.rectangle(face_frame,(squaresize[0],squaresize[1]),(squaresize[0]+squaresize[2],squaresize[1]+squaresize[3]),(0,255,255),3)
    cv2.rectangle(infr_frame,(squaresize[0],squaresize[1]),(squaresize[0]+squaresize[2],squaresize[1]+squaresize[3]),(0,255,255),3)


    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    archive_frame = (np.reshape(archive, mlx_shape))
    archive_frame = np.flipud(archive_frame)
    archive_scale = cv2.resize(archive_frame, (480, 640))
    archive_scale = archive_scale[0:480, 70:680]
    
    for (x, y, w, h) in faces:

        cv2.rectangle(face_frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
        cv2.rectangle(infr_frame, (x-40,y+20), (x+w-40, y+h+20), (255, 0, 0), 2)
        
        if archive_scale[x-40:x+w-40, y+20:y+h+20] is None:
            continue
        try:
            human_temp = cv2.resize(archive_scale[x-40:x+w-40, y:y+h], (100,100)) + delta

            cv2.putText(face_frame, str("%.2f" % round(np.average(human_temp),2)), (int(x + w/2), int(y + h/2)), \
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)    

            
                
        except cv2.error as e:
            continue

    
    

    cv2.putText(face_frame, str("Delta:" + ("%.2f" % round(delta,2))), (150,150), \
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('frame', face_frame)
    cv2.imshow('infra', infr_frame)

    

    print('Delta', delta, '*C')






    if cv2.waitKey(1) == ord('q'):
        break

    if keyboard.is_pressed('w'):
        delta = 0.0
        persist = [0,0,0]

    if keyboard.is_pressed('e'):
        # cannot = False
        retry = True
        delta = 0.0

    

    conn.send('good'.encode())


    end = time.time()

    # print(1/ (end - start), 'fps')

    if fps_counter.__len__() < 10:
        fps_counter.append(1 / (end - start))

    else:
        fps_counter.popleft()
        fps_counter.append(1 / (end - start))
        print(np.average(fps_counter), 'fps')

        

    



# close connection and exit
conn.close()
cap.release()
# break 
cv2.destroyAllWindows()