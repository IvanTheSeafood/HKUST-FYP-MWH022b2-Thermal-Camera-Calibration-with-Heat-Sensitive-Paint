import socket
import numpy as np
import adafruit_mlx90640
import time,board,busio
import pickle
import struct

ip = '192.169.137.1'
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((ip, 8080))

def send_msg(s, msg):
    msg = struct.pack('>I', len(msg)) + msg.pack
    s.sendall(msg)

print("Client connected")

while True:
    i2c = busio.I2C(board.SCL, board.SDA, frequency=1000000)
    mlx = adafruit_mlx90640.MLX90640(i2c)
    mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_8_HZ
    mlx_shape = (24,32)

    frame = np.zeros((24*32))

    try:
        mlx.getFrame(frame)
    
    except ValueError:
        continue
    
    msg = pickle.dumps(frame)
    send_msg(client, msg)

    reply = client.resv(2048)
    print(reply.decode())

client.close()