# Face Recognition Based Attendance System

This project implements an automated attendance system using face recognition. The system identifies faces in real-time through a webcam and automatically marks attendance for recognized individuals.

## Features

- Automatic face recognition for attendance marking
- Real-time face detection with a rolling buffer for confirmation
- Option to add, update, and remove students' face images
- Attendance logs stored in CSV format
- Simple GUI to input lecture details (name, teacher, time slot)
- GPU support for face recognition (if available)

## Requirements

The system relies on the following Python libraries:

- **OpenCV**: For video capture and image processing
- **dlib**: For face detection and recognition
- **face_recognition**: For encoding and recognizing faces
- **numpy**: For array and matrix operations
- **tkinter**: For GUI elements to input lecture details
- **csv**: To manage attendance records
- **pickle**: For caching face encodings and metadata
- **datetime**: For timestamping attendance
- **collections**: For using `defaultdict`

To install the required dependencies, create a virtual environment and install the packages using `pip`.

## Project Setup

### Directory Structure

- **Training_images/**: Store student images (e.g., `John_Doe.jpg`) used for training face recognition.
- **Attendance/**: Contains the attendance logs in CSV format.
- **face_encodings_cache.pkl**: A cache file that stores face encodings and names.
- **face_metadata.pkl**: A cache file for metadata about face recognition images.

### Preparing Training Data

Ensure that you have students' images stored in the `Training_images/` directory. Each student's image should be named after them (e.g., `John_Doe.jpg`). The system will extract face encodings from these images for recognition.

### Running the System

To run the system, execute the script and provide the lecture details when prompted. The system will then start capturing video, detect faces, and mark attendance automatically.

### Attendance Marking

Attendance is stored in a CSV file within the `Attendance/` directory. Each file is named according to the lecture and date, for example: `Attendance_<Lecture Name>_<Date>.csv`. The file contains the following details for each student:

- Student's name
- Time of attendance
- Lecture name
- Teacher's name
- Time slot
- Date of the lecture

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Acknowledgments

- The face recognition functionality uses the [face_recognition](https://github.com/ageitgey/face_recognition) library.
- Face detection is powered by [dlib](http://dlib.net/).
