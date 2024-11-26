import cv2
import numpy as np
import face_recognition
import os
import pickle
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, ttk
import csv
import dlib
from collections import defaultdict

class AttendanceSystem:
    def __init__(self):
        self.path = 'Training_images'
        self.images = []
        self.classNames = []
        self.lecture_info = None
        self.running = True
        self.cache_file = 'face_encodings_cache.pkl'
        self.metadata_file = 'face_metadata.pkl'
        # Parameters for reducing false positives
        self.recognition_threshold = 0.5  # Lower number = stricter matching
        self.consecutive_frames_required = 3  # Number of frames needed for confirmation
        self.face_recognition_buffer = defaultdict(list)  # Buffer for tracking face recognitions
        self.buffer_size = 10  # Size of the rolling buffer for each face
        # Add new parameters for optimization
        self.process_every_n_frames = 8  # Only process every nth frame
        self.frame_counter = 0
        self.frame_size = (640, 480)  # Reduce frame size for processing
        self.detection_scale = 0.25    # Scale for face detection

        
        # Create Training_images directory if it doesn't exist
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            print(f"Created directory: {self.path}")
            print("Please add training images to this directory before running the system.")
            
        self.load_training_images()

    def load_training_images(self):
        """Load training images and identify new images that need encoding."""
        myList = os.listdir(self.path)
        
        # Load existing metadata if available
        existing_metadata = self.load_metadata()
        new_files = []
        removed_files = []
        
        # Check for new and removed files
        current_files = set(myList)
        if existing_metadata:
            old_files = set(existing_metadata['files'])
            new_files = list(current_files - old_files)
            removed_files = list(old_files - current_files)
        else:
            new_files = list(current_files)
            
        # Load all current images and names
        self.images = []  # Clear existing images
        self.classNames = []  # Clear existing names
        for cl in myList:
            curImg = cv2.imread(f'{self.path}/{cl}')
            self.images.append(curImg)
            self.classNames.append(os.path.splitext(cl)[0])
            
        print('Total students:', self.classNames)
        if new_files:
            print('New students detected:', [os.path.splitext(f)[0] for f in new_files])
        if removed_files:
            print('Removed students:', [os.path.splitext(f)[0] for f in removed_files])
            
        return new_files, removed_files

    def save_metadata(self, files):
        """Save metadata about processed files."""
        metadata = {
            'files': files,
            'last_updated': datetime.now().isoformat()
        }
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(metadata, f)

    def load_metadata(self):
        """Load metadata about previously processed files."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'rb') as f:
                return pickle.load(f)
        return None

    def save_encodings_to_cache(self, encodeListKnown):
        """Save the encodings and class names to a cache file."""
        with open(self.cache_file, 'wb') as f:
            pickle.dump({
                'encodings': encodeListKnown,
                'names': self.classNames
            }, f)
        # Save the current state of files
        self.save_metadata(os.listdir(self.path))
        print("Encodings and metadata saved to cache.")

    def load_encodings_from_cache(self):
        """Load the encodings and class names from the cache file, if available."""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                data = pickle.load(f)
                return data['encodings']
        return None

    def findEncodings(self, images, new_files=None):
        """Find the face encodings for the given images."""
        print("CUDA available:", dlib.DLIB_USE_CUDA)
        
        # If we have existing encodings and only need to process new files
        existing_encodings = self.load_encodings_from_cache()
        if existing_encodings and new_files:
            print(f"Processing {len(new_files)} new faces...")
            new_encodings = []
            file_names = os.listdir(self.path)
            
            # Only process new files
            for file_name in new_files:
                img_path = os.path.join(self.path, file_name)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                try:
                    encodes = face_recognition.face_encodings(img)
                    if encodes:
                        new_encodings.append(encodes[0])
                        print(f"Processed new face: {os.path.splitext(file_name)[0]}")
                    else:
                        print(f"Warning: No face found in {file_name}")
                except Exception as e:
                    print(f"Error processing {file_name}: {str(e)}")
            
            # Combine existing and new encodings
            return existing_encodings + new_encodings
        
        # If no cache exists or complete reprocessing is needed
        print("Processing all faces...")
        encodeList = []
        for i, img in enumerate(images):
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encodes = face_recognition.face_encodings(img)
                if encodes:
                    encodeList.append(encodes[0])
                else:
                    print(f"Warning: No face found in {self.classNames[i]}")
            except Exception as e:
                print(f"Error processing {self.classNames[i]}: {str(e)}")
        return encodeList

    def update_encodings(self):
        """Update encodings based on changes in the training folder."""
        new_files, removed_files = self.load_training_images()
        
        if not (new_files or removed_files):
            print("No changes detected in training images.")
            return self.load_encodings_from_cache()
        
        print("Changes detected in training images. Updating encodings...")
        # If files were removed, we need to do a complete reprocessing
        if removed_files:
            print("Removed files detected. Performing complete reprocessing...")
            encodeListKnown = self.findEncodings(self.images)
        else:
            # If only new files were added, we can process just those
            encodeListKnown = self.findEncodings(self.images, new_files)
            
        self.save_encodings_to_cache(encodeListKnown)
        return encodeListKnown

    def process_recognition(self, name, confidence):
        """
        Process face recognition results using a rolling buffer system.
        Returns True if the face is confirmed, False otherwise.
        """
        # Add new recognition to buffer
        self.face_recognition_buffer[name].append(confidence)
        
        # Keep buffer at specified size
        if len(self.face_recognition_buffer[name]) > self.buffer_size:
            self.face_recognition_buffer[name].pop(0)
        
        # Check if we have enough consistent recognitions
        if len(self.face_recognition_buffer[name]) >= self.consecutive_frames_required:
            recent_recognitions = self.face_recognition_buffer[name][-self.consecutive_frames_required:]
            # Check if all recent recognitions are confident enough
            if all(conf <= self.recognition_threshold for conf in recent_recognitions):
                return True
        return False

    def get_lecture_info(self):
        """Get lecture information from the user via a GUI."""
        root = tk.Tk()
        root.title("Lecture Information")
        root.geometry("400x400")
        root.configure(bg='#f0f0f0')
        root.eval('tk::PlaceWindow . center')

        style = ttk.Style()
        style.configure('TCombobox', 
                       background='white',
                       fieldbackground='white')

        # Frame for all content
        main_frame = tk.Frame(root, bg='#f0f0f0')
        main_frame.pack(expand=True, fill='both', padx=20, pady=20)

        # Title
        title_label = tk.Label(main_frame, 
                             text="Attendance System", 
                             font=('Arial', 20, 'bold'),
                             bg='#f0f0f0')
        title_label.pack(pady=(0, 20))

        # Lecture Name
        tk.Label(main_frame, text="Lecture Name:", 
                font=('Arial', 12),
                bg='#f0f0f0').pack()
        lecture_entry = tk.Entry(main_frame, 
                               width=30,
                               font=('Arial', 11))
        lecture_entry.pack(pady=(5, 15))

        # Teacher Name
        tk.Label(main_frame, text="Teacher Name:", 
                font=('Arial', 12),
                bg='#f0f0f0').pack()
        teacher_entry = tk.Entry(main_frame, 
                               width=30,
                               font=('Arial', 11))
        teacher_entry.pack(pady=(5, 15))

        # Time Slots
        tk.Label(main_frame, text="Lecture Slot:", 
                font=('Arial', 12),
                bg='#f0f0f0').pack()

        time_slots = [
            "8:00 AM - 9:00 AM",
            "9:00 AM - 10:00 AM",
            "10:00 AM - 11:00 AM",
            "11:00 AM - 12:00 PM",
            "12:00 PM - 1:00 PM",
            "2:00 PM - 3:00 PM",
            "3:00 PM - 4:00 PM",
            "4:00 PM - 5:00 PM"
        ]

        slot_combobox = ttk.Combobox(main_frame, 
                                    values=time_slots, 
                                    width=27,
                                    font=('Arial', 11))
        slot_combobox.set("Select Time Slot")
        slot_combobox.pack(pady=(5, 25))

        def submit():
            lecture_name = lecture_entry.get().strip()
            teacher_name = teacher_entry.get().strip()
            time_slot = slot_combobox.get()
            
            if not lecture_name or not teacher_name or time_slot == "Select Time Slot":
                messagebox.showerror("Error", "Please fill in all fields!")
                return
                
            self.lecture_info = {
                'lecture': lecture_name,
                'teacher': teacher_name,
                'slot': time_slot,
                'date': datetime.now().strftime('%Y-%m-%d')
            }
            root.destroy()

        # Buttons Frame
        buttons_frame = tk.Frame(main_frame, bg='#f0f0f0')
        buttons_frame.pack(pady=10)

        submit_button = tk.Button(
            buttons_frame, 
            text="Start Attendance", 
            command=submit,
            bg='#4CAF50', 
            fg='white', 
            width=20,
            height=2,
            font=('Arial', 11, 'bold'),
            cursor='hand2'
        )
        submit_button.pack(pady=5)
        
        cancel_button = tk.Button(
            buttons_frame,
            text="Exit Program",
            command=lambda: self.exit_program(root),
            bg='#f44336',
            fg='white',
            width=20,
            height=1,
            font=('Arial', 11),
            cursor='hand2'
        )
        cancel_button.pack(pady=5)
        
        self.root = root
        root.mainloop()

    def exit_program(self, root):
        """Exit the program."""
        self.running = False
        root.destroy()

    def markAttendance(self, name):
        """Mark attendance in the CSV file."""
        # Create 'Attendance' directory if it doesn't exist
        attendance_dir = 'Attendance'
        if not os.path.exists(attendance_dir):
            os.makedirs(attendance_dir)
            
        filename = os.path.join(attendance_dir, 
                              f"Attendance_{self.lecture_info['lecture']}_{self.lecture_info['date']}.csv")
        
        if not os.path.exists(filename):
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Name', 'Time', 'Lecture', 'Teacher', 'Slot', 'Date'])
        
        with open(filename, 'r+', newline='') as f:
            myDataList = list(csv.reader(f))
            nameList = [row[0] for row in myDataList]
            
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                writer = csv.writer(f)
                writer.writerow([name, dtString, self.lecture_info['lecture'], 
                               self.lecture_info['teacher'], self.lecture_info['slot'], 
                               self.lecture_info['date']])
                print(f"Marked attendance for {name}")

    def draw_controls_info(self, img):
        """Draw controls info on the camera frame."""
        overlay = img.copy()
        cv2.rectangle(overlay, (10, img.shape[0]-90), (300, img.shape[0]-10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
        cv2.putText(img, "Controls:", (20, img.shape[0]-65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, "Press 'q' to quit program", (20, img.shape[0]-40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, "Press 'c' to change lecture", (20, img.shape[0]-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def run_camera(self):
        """Run the camera feed with optimized face recognition and attendance marking."""
        print('Checking for updates in training images...')
        encodeListKnown = self.update_encodings()
        
        if encodeListKnown is None:
            print('No encodings found. Starting fresh encoding...')
            encodeListKnown = self.findEncodings(self.images)
            self.save_encodings_to_cache(encodeListKnown)
        
        print('Encoding Complete')

        cap = cv2.VideoCapture(0)
        # Set optimal camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_size[1])
        cap.set(cv2.CAP_PROP_FPS, 30)  # Set to 30 FPS
        
        # Dictionary to track attendance status
        attendance_marked = set()
        
        # Variables for face tracking
        prev_face_locs = []
        prev_names = []
        prev_confidences = []

        while True:
            success, img = cap.read()
            if not success:
                print("Failed to grab frame")
                break

            # Only process every nth frame for face detection
            if self.frame_counter % self.process_every_n_frames == 0:
                # Resize image for faster processing
                imgS = cv2.resize(img, (0, 0), None, self.detection_scale, self.detection_scale)
                imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

                # Use CNN model only when GPU is available, otherwise use HOG
                if dlib.DLIB_USE_CUDA:
                    facesCurFrame = face_recognition.face_locations(imgS, model="cnn")
                else:
                    facesCurFrame = face_recognition.face_locations(imgS, model="hog")
                
                encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

                # Reset previous results
                prev_face_locs = []
                prev_names = []
                prev_confidences = []

                for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                    matches = face_recognition.compare_faces(encodeListKnown, encodeFace, 
                                                          tolerance=self.recognition_threshold)
                    faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                    
                    if len(faceDis) > 0:
                        matchIndex = np.argmin(faceDis)
                        min_distance = faceDis[matchIndex]
                        
                        if matches[matchIndex] and min_distance < self.recognition_threshold:
                            name = self.classNames[matchIndex].upper()
                            
                            if self.process_recognition(name, min_distance):
                                y1, x2, y2, x1 = faceLoc
                                # Scale back to full size
                                y1, x2, y2, x1 = [int(coord * (1/self.detection_scale)) for coord in [y1, x2, y2, x1]]
                                
                                prev_face_locs.append((x1, y1, x2, y2))
                                prev_names.append(name)
                                prev_confidences.append(min_distance)
                                
                                if name not in attendance_marked:
                                    self.markAttendance(name)
                                    attendance_marked.add(name)
                                    print(f"Attendance marked for {name}")
            
            # Draw results from previous successful detection
            for face_loc, name, confidence in zip(prev_face_locs, prev_names, prev_confidences):
                x1, y1, x2, y2 = face_loc
                confidence_score = (1 - confidence) * 100
                
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, f"{name} ({confidence_score:.1f}%)", 
                          (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 
                          0.6, (255, 255, 255), 2)

            # Display lecture info and controls
            self.draw_lecture_info(img)
            self.draw_controls_info(img)
            
            cv2.imshow('Attendance System', img)
            self.frame_counter += 1
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
                break
            elif key == ord('c'):
                break

        cap.release()
        cv2.destroyAllWindows()


    def draw_lecture_info(self, img):
        """Draw lecture information on the frame."""
        # Create semi-transparent overlay for better text readability
        overlay = img.copy()
        cv2.rectangle(overlay, (5, 5), (400, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
        
        cv2.putText(img, f"Lecture: {self.lecture_info['lecture']}", 
                   (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Teacher: {self.lecture_info['teacher']}", 
                   (10, 60), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Slot: {self.lecture_info['slot']}", 
                   (10, 90), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)

    def run(self):
        """Main loop to get lecture info and run camera."""
        while self.running:
            self.get_lecture_info()
            if not self.running:
                print("Program terminated")
                break
                
            if not self.lecture_info:
                print("Attendance system cancelled")
                break

            self.run_camera()

if __name__ == "__main__":
    attendance_system = AttendanceSystem()
    attendance_system.run()