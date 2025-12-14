# 🎭 Face Mask Detection & Recognition System

This project is a real-time face recognition and mask detection system developed using **OpenCV**, **TensorFlow/Keras**, and **PyQt5**. The system authenticates users by recognizing their faces and checks whether they are wearing a mask. When a registered user is detected without a mask, the system automatically logs the event and captures a photo.

## 🚀 Features

-   **User Registration (Sign Up):** Users can register with both maskless and masked face data.
-   **Biometric Login (Sign In):** Secure login using facial recognition powered by **FaceNet** (deep learning).
-   **Real-Time Mask Detection:** Instant mask checking via live camera feed.
-   **Violation Tracking & Logging:** If a logged-in user is detected without a mask:
    -   Date and time are recorded in `users/maskless_log.csv`.
    -   A photo of the user is saved to the `users/maskless_photos/` directory.
    -   To prevent spam, logs are taken at most once every 3 minutes per user.
-   **Modern Interface:** A user-friendly and stylish interface developed with **PyQt5**.

## 🛠️ Requirements

You need the following Python libraries to run this project:

*   Python 3.x
*   OpenCV (`cv2`)
*   TensorFlow / Keras
*   keras-facenet
*   PyQt5
*   NumPy

## 📦 Installation

1.  **Clone the Project:**
    ```bash
    git clone https://github.com/berhankokum/face-mask-detection.git
    cd face-mask-detection
    ```

2.  **Install Required Libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    Alternatively, you can install them manually:
    ```bash
    pip install opencv-python tensorflow keras-facenet PyQt5 numpy
    ```

## 💻 Usage

Run the `gui.py` file in the main directory to start the application:

```bash
python gui.py
```

### 1. Sign Up
1.  Enter your username and click the **"sign up"** button or the **"Start Recording"** button (which appears after clicking sign up).
2.  The system first scans your **maskless** face.
3.  Then it scans your **masked** face and completes the registration.

### 2. Sign In
1.  Click the **"sign in"** button.
2.  Look at the camera; the system will automatically log you in when it recognizes you.

### 3. Mask Control
1.  After logging in, the system continuously monitors your face.
2.  If you are wearing a mask, a **Green** frame and "Masked" text appear on the screen.
3.  If you are not wearing a mask, a **Red** frame and "No Mask" text appear. In this case, the system automatically takes your photo and records it in the log file.

## 📂 Project Structure

```
face-mask-detection/
├── face_detector/           # Caffe models for face detection
│   ├── deploy.prototxt
│   └── res10_300x300_ssd...
├── users/                   # User data and logs (Automatically created)
│   ├── maskless_photos/     # Photos of users caught without masks
│   ├── maskless_log.csv     # Violation records
│   └── {username}.pkl       # User face embedding data
├── gui.py                   # Main application and UI code
├── mask_detector_model.h5   # Trained mask detection model
└── README.md                # Project documentation
```

## 🧠 Technical Details

*   **Face Detection:** Uses OpenCV's DNN module and a Caffe-based SSD model.
*   **Face Recognition:** **keras-facenet** library is used to extract 128/512-dimensional embedding vectors from faces, matching them using Euclidean distance.
*   **Mask Detection:** A pre-trained CNN model (`mask_detector_model.h5`) is used to classify whether the face is masked or not.

---
*Developer: Berhan Köküm*
