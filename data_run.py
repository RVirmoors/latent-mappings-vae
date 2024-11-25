import cv2
import mediapipe as mp
import numpy as np
import time
import argparse
import time
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *

from pythonosc import udp_client

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

pTime = 0
cTime = 0

pose_info=[]
filename = "data.csv"

parser = argparse.ArgumentParser()
parser.add_argument("--ip", default="127.0.0.1", help="The ip of the OSC server")
parser.add_argument("--port", type=int, default=8888, help="The port the OSC server is listening on")
parser.add_argument("--run", type=bool, default=False, help="Run inference. If false, just record data.")
args = parser.parse_args()

client = udp_client.SimpleUDPClient(args.ip, args.port)
vae = VAE(input_dim, hidden_dim, latent_dim)
if not args.run:
    # Load the saved state dictionary into the model
    vae.load_state_dict(torch.load("mocap_vae.pth"))
    vae.eval()  # Set the model to evaluation mode
    print("Model loaded!")


def normalize_latent_space(mean, logvar):
    z = vae.reparameterize(mean, logvar)
    z_min, z_max = -2, 2  # ±2 std dev
    return ((z - z_min) / (z_max - z_min)).clamp(0, 1)

# For webcam input:a
cap = cv2.VideoCapture(0)
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if results.pose_landmarks:
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x *w), int(lm.y*h)
                    cv2.circle(image, (cx,cy), 3, (255,0,255), cv2.FILLED)
                    pose_info.append(cx/cap.get(3))
                    pose_info.append(cy/cap.get(4))

                
                if args.run:
                    with torch.no_grad():
                        mocap_input = torch.tensor(pose_info)
                        mean, logvar = vae.encode(mocap_input)
                        normalized_params = normalize_latent_space(mean, logvar)
                    client.send_message("/synth", normalized_params.tolist() )
                else:
                    writer.writerow(pose_info)
                pose_info = []
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                        
            cTime = time.time()
            fps = 1/(cTime-pTime)
            pTime = cTime

            cv2.putText(image,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

            cv2.imshow("Image", image)

            # ESC interrupt
            if cv2.waitKey(5) & 0xFF == 27:
                break
