import os
import cv2
import numpy as np
import threading
import asyncio
import websockets
from mediapipe.python.solutions import holistic as mp_holistic
from mediapipe.python.solutions import drawing_utils as mp_drawing
from sagemaker import Session
from sagemaker.tensorflow.model import TensorFlowPredictor
import pygame
pygame.init()

# Determine the absolute path to the project folder
project_folder = os.path.dirname(os.path.abspath(__file__))

# Define the path to the audio files within the project folder
audio_folder = os.path.join(project_folder, 'sounds')

# Your other code remains the same

# Create a list to store connected WebSocket clients
connected_clients = set()
toberemoved = set()
t = 1

# Use the audio_folder variable to construct the file paths for audio files
prediction_audio_map = {
    "salam": os.path.join(audio_folder, "salam.mp3"),
    "labasse": os.path.join(audio_folder, "labasse.mp3"),
    "kidayre": os.path.join(audio_folder, "kidayr.mp3"),
    "chokran": os.path.join(audio_folder, "chokran.mp3"),
    "ana": os.path.join(audio_folder, "ana.mp3"),
    "farhane": os.path.join(audio_folder, "farhane.mp3"),
    "hitache": os.path.join(audio_folder, "hitache.mp3"),
    "ah": os.path.join(audio_folder, "ah.mp3"),
    "jihaz": os.path.join(audio_folder, "jihaz.mp3"),
    "sahal leia": os.path.join(audio_folder, "sahel lia.mp3"),
    "tawasole": os.path.join(audio_folder, "tawasol.mp3"),
}
# Add quotes around file paths
for key, value in prediction_audio_map.items():
    prediction_audio_map[key] = f'"{value}"'

# Load all the audio tracks into a dictionary
# loaded_audio = {}
# for key, path in prediction_audio_map.items():
#     loaded_audio[key] = AudioSegment.from_mp3(path)

def play_audio( selected_track):
    audio_path=prediction_audio_map[selected_track].strip('"')
    pygame.mixer.music.load(audio_path)
    pygame.mixer.music.play()

    pass

def set_t(tr):
    global t
    t = tr

def get_t():
    global t

    return t

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False                 
    results = model.process(image)               
    image.flags.writeable = True                  
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right

    

def draw_styled_landmarks(image, results):

    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

    

colors = [
    (245, 117, 16), (117, 245, 16), (16, 117, 245), (245, 16, 117), (117, 16, 245),
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (128, 128, 128), (64, 0, 0), (0, 64, 0),
    (0, 0, 64)
]

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame


def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([ lh, rh])

sequence = []
sentence = []
threshold = 0.9
actions = np.array(["salam", "labasse", "kidayre", "chokran", "ana", "farhane", "hitache", "ah", "jihaz", "sahal leia", "tawasole"])

no_sequences = 30
sequence_length = 30
num_classes = len(actions)


RTSP_URL = 'rtsp://192.168.11.210:5000/unicast'

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

# Create a SageMaker predictor by specifying the endpoint
endpoint_name = 'EchosignLive'  # Replace with your endpoint name
sagemaker_session = Session()
predictor = TensorFlowPredictor(endpoint_name, sagemaker_session)

# Create a lock to ensure safe access to shared resources
lock = threading.Lock()

# Define a WebSocket handler function
async def video_stream(websocket, path):
    # Send video frames to the connected client
    try:
        while True:
            if not connected_clients:
                await asyncio.sleep(1)  # Sleep for 1 second if no clients are connected
                continue
            
            # Your video processing code here (same as in your previous code)
            # ...
            
            # Encode the frame as JPEG
            _, buffer = cv2.imencode('.jpg', image)
            image_bytes = buffer.tobytes()

            # Send the frame to all connected clients
            for client in connected_clients:
                await client.send(image_bytes)
    except websockets.exceptions.ConnectionClosedError:
        pass  # Handle client disconnection

def predict_and_update_sequence():
    global sequence
    global sentence

    with lock:
        if len(sequence) == sequence_length:
            res = predictor.predict(np.expand_dims(sequence, axis=0))["predictions"][0]
            print(actions[np.argmax(res)])

            if res[np.argmax(res)] > threshold:
                play_audio(actions[np.argmax(res)])
                if len(sentence) > 0:
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5:
                sentence = sentence[-5:]

            sequence = []
skip_frames = 5  # Adjust as needed

async def video_processing(client):
    # 1. New detection variables
    global sequence 
    global sentence 
    global actions
    global sequence_length 
    sequence = []
    sentence = []

   
    frm,lmt, frame_count = (0,10,0)
    RTSP_URL = 'rtsp://192.168.11.210:5000/unicast'

    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
    #RTSP_URL, cv2.CAP_FFMPEG
    cap = cv2.VideoCapture(0)
    print("Video processing started")
    while True:
        # if not connected_clients:
        #     print("No clients connected")
        #     await  asyncio.sleep(1)  # Sleep for 1 second if no clients are connected
        #     continue
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                if get_t() != 0:
                    cap.release()
                    cv2.destroyAllWindows()
                    connected_clients.clear()
                    set_t(1)
                    return
                ret, frame = cap.read()
                frame_count += 1
               
                if frame_count % skip_frames != 0:
                    continue  # Skip frames
                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)

                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-sequence_length:]
                frm += 1
                if frm>=lmt and len(sequence) == sequence_length:
                    predict_and_update_sequence()
                    frm = 0

                cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, ' '.join(sentence), (3, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                # Encode the frame as JPEG
                _, buffer = cv2.imencode('.jpg', image)
                image_bytes = buffer.tobytes()

                # Send the frame to all connected clients
                # if toberemoved:
                #     for client in toberemoved:
                #         connected_clients.remove(client)
                #     toberemoved.clear()
                #     continue
                
                try:
                    
                        await client.send(image_bytes)
                except websockets.exceptions.ConnectionClosedError:
                    
                    print("Client erroooor")
                    cap.release()
                    cv2.destroyAllWindows()
                    connected_clients.clear()
                    set_t(1)
                    return

                            
            
    
                    
async def client_handler(websocket, path):
    global connected_clients
    
    print("Client connected")
    # Add the client to the list of connected clients
    connected_clients.add(websocket)
    while get_t() == 0:
        await asyncio.sleep(1)
    if get_t() == 1:
        set_t(0)
        await video_processing(websocket)
    
    try:
        await websocket.wait_closed()  # Wait for the WebSocket connection to close
    finally:
        print("Client disconnected")
        set_t(1)
                # Remove the client from the list of connected clients when the connection is closed
    



# Start the video processing in one thread
start_server = websockets.serve(client_handler, "localhost", 8765)

async def run_video_processing():
    await asyncio.gather(start_server, asyncio.create_task(video_processing()))
event_loop = asyncio.new_event_loop()  # Create a new event loop

  # Set the event loop for this thread
print("Server starting")
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
print("Video processing started")
# asyncio.get_event_loop().run_until_complete(start_server)
# asyncio.get_event_loop().run_forever()
print("Server started")
# You can continue with any other code or functionality as needed.
