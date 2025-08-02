import os
import cv2
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog, messagebox

def predict_on_video(video_file_path, output_file_path, SEQUENCE_LENGTH, model_path, IMAGE_HEIGHT, IMAGE_WIDTH, CLASSES_LIST):
    model = load_model(model_path)
    video_reader = cv2.VideoCapture(video_file_path)
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(*'mp4v'), 
                                   video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)
    predicted_class_name = ''

    while video_reader.isOpened():
        ok, frame = video_reader.read() 
        if not ok:
            break
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255.0
        frames_queue.append(normalized_frame)
        if len(frames_queue) == SEQUENCE_LENGTH:
            predicted_labels_probabilities = model.predict(np.expand_dims(frames_queue, axis=0))[0]
            predicted_label = np.argmax(predicted_labels_probabilities)
            predicted_class_name = CLASSES_LIST[predicted_label]
        cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        video_writer.write(frame)
    video_reader.release()
    video_writer.release()

def select_video_file():
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])
    if file_path:
        video_file_path.set(file_path)

def select_output_folder():
    folder_path = filedialog.askdirectory()
    if folder_path:
        output_folder_path.set(folder_path)

def run_action_recognition():
    video_path = video_file_path.get()
    output_folder = output_folder_path.get()
    output_path = os.path.join(output_folder, "output.mp4")
    SEQUENCE_LENGTH = 20
    model_path = r"C:\Users\DELL 7490\Desktop\Cantilever\Python Scripts\LRCN_model___Date_Time_2024_06_12__12_38_06___Loss_0.40667715668678284___Accuracy_0.9098360538482666.keras"
    IMAGE_HEIGHT = 64 
    IMAGE_WIDTH = 64   
    CLASSES_LIST = ["WalkingWithDog", "TaiChi", "Swing", "HorseRace"]  
    try:
        predict_on_video(video_path, output_path, SEQUENCE_LENGTH, model_path, IMAGE_HEIGHT, IMAGE_WIDTH, CLASSES_LIST)
        messagebox.showinfo("Success", f"Action recognition complete. Output saved to {output_path}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Create the main window
root = tk.Tk()
root.title("Action Recognition")
#root.geometry("800x600")  # Set the initial size of the window
root.state('zoomed')
# Create UI elements
canvas = tk.Canvas(root, width=640, height=480)  # Set canvas size to display video
canvas.pack()


# Create and set variables
video_file_path = tk.StringVar()
output_folder_path = tk.StringVar()

# Create UI elements
video_label = tk.Label(root, text="Select video file:")
video_label.pack()
video_entry = tk.Entry(root, textvariable=video_file_path, width=50)
video_entry.pack()
video_button = tk.Button(root, text="Browse", command=select_video_file)
video_button.pack()

output_label = tk.Label(root, text="Select output folder:")
output_label.pack()
output_entry = tk.Entry(root, textvariable=output_folder_path, width=50)
output_entry.pack()
output_button = tk.Button(root, text="Browse", command=select_output_folder)
output_button.pack()


run_button = tk.Button(root, text="Run Action Recognition", command=run_action_recognition)
run_button.pack()

# Start the GUI event loop
root.mainloop()
