# Lane-Detection-using-U-Net-with-Spatiotemporal


This repository contains implementations for lane detection using both traditional computer vision techniques and deep learning models. The project explores various U-Net architectures, including a version enhanced with ConvLSTM layers to incorporate spatiotemporal information for more robust video-based lane detection.

## Features

*   **Classical Computer Vision Approach**: Lane detection using OpenCV, featuring color space transformation, perspective warping, and polynomial fitting.
*   **U-Net with Binary Focal Loss**: A U-Net model trained for lane segmentation using Binary Focal Loss, which is effective for handling class imbalance.
*   **U-Net with ConvLSTM and IoU Loss**: An advanced U-Net model incorporating `ConvLSTM2D` layers at its bottleneck to leverage temporal data from video sequences. This model is trained using an IoU (Intersection over Union) based loss function.
*   **Training and Inference Scripts**: Includes scripts to train the models from scratch and to run inference on single images and video files.

## Repository Structure

```
.
├── Damoc_Robert-Marian.pptx      # Project presentation
├── Damoc_Robert-Marian_licenta.docx # Project thesis document
├── cv.py                         # Classical computer vision lane detection implementation
├── test_image.py                 # Script to test a trained model on a single image
├── video_unet.py                 # Script to run lane detection on a video using a U-Net model
├── unet_binaryfocal/
│   ├── lane_cv_dropout_batch.h5  # Pre-trained U-Net model (Focal Loss)
│   └── run_unet.py               # Training script for the U-Net with Binary Focal Loss
├── unet_iou_loss/
│   └── lane_cv_dropout_batch.h5  # Pre-trained U-Net model (IoU Loss)
└── unet_lstm/
    └── run_unet.py               # Training script for the U-Net + ConvLSTM model with IoU Loss
```

## Models

### 1. Classical CV Lane Detection (`cv.py`)
This method uses a pipeline of traditional image processing techniques:
1.  **Preprocessing**: Converts the image to grayscale and HSV color spaces to create a binary mask for yellow and white lane lines.
2.  **Perspective Transform**: Warps the detected lane lines into a bird's-eye view.
3.  **Lane Fitting**: Uses a sliding window approach on a histogram of the warped image to identify lane pixels and fits a second-degree polynomial to each lane line.
4.  **Visualization**: Unwarps the detected lane area and overlays it on the original frame, displaying the calculated radius of curvature.

### 2. U-Net with Binary Focal Loss (`unet_binaryfocal/`)
This is a standard U-Net architecture designed for semantic segmentation.
-   **Architecture**: Consists of a contracting path (encoder) to capture context and a symmetric expanding path (decoder) for precise localization.
-   **Loss Function**: Uses `BinaryFocalLoss` to address the imbalance between the lane pixels and the background.
-   **Training**: The `run_unet.py` script uses a data generator to feed images and corresponding masks to the model.
![Accuracy](https://github.com/user-attachments/assets/65905bbd-3a0a-463b-9a96-c32a39d4193b)
![Compared](<img width="892" height="198" alt="Picture1" src="https://github.com/user-attachments/assets/19bd1bd0-edb0-4ec7-83f2-4107568f7cc9" />
)
![Loss](<img width="640" height="480" alt="loss_plot_old" src="https://github.com/user-attachments/assets/bb103422-44d2-4d46-a567-f01b18250b58" />
)
### 3. U-Net with ConvLSTM (`unet_lstm/`)
This model enhances the standard U-Net by adding `ConvLSTM2D` layers in the bottleneck. This allows the model to learn spatiotemporal features from sequential frames in a video, improving temporal consistency.
-   **Architecture**: Integrates two `ConvLSTM2D` layers between the encoder and decoder paths.
-   **Loss Function**: Utilizes an `iou_loss` (1 - IoU), which directly optimizes the Intersection over Union metric. This is highly effective for segmentation tasks.
-   **Metric**: The model is evaluated using the `iou` metric.
![Accuracy IoU Loss]([[https://github.com/tograh/testrepository/3DTest.png](https://github.com/doilonar/Lane-Detection-using-U-Net-with-Spatiotemporal/blob/main/unet_binaryfocal/accuracy_plot_old.png)](https://github.com/doilonar/Lane-Detection-using-U-Net-with-Spatiotemporal/blob/main/unet_iou_loss/accuracy_plot_iou.png))
![Compared IoU Loss]([[[https://github.com/tograh/testrepository/3DTest.png](https://github.com/doilonar/Lane-Detection-using-U-Net-with-Spatiotemporal/blob/main/unet_binaryfocal/accuracy_plot_old.png)](https://github.com/doilonar/Lane-Detection-using-U-Net-with-Spatiotemporal/blob/main/unet_binaryfocal/Picture1.png)](https://github.com/doilonar/Lane-Detection-using-U-Net-with-Spatiotemporal/blob/main/unet_iou_loss/Picture2.png))
![Loss IoU Loss]([[[https://github.com/tograh/testrepository/3DTest.png](https://github.com/doilonar/Lane-Detection-using-U-Net-with-Spatiotemporal/blob/main/unet_binaryfocal/accuracy_plot_old.png)](https://github.com/doilonar/Lane-Detection-using-U-Net-with-Spatiotemporal/blob/main/unet_binaryfocal/loss_plot_old.png)](https://github.com/doilonar/Lane-Detection-using-U-Net-with-Spatiotemporal/blob/main/unet_iou_loss/loss_plot.png))

![Accuracy LSTM Integration]([[[https://github.com/tograh/testrepository/3DTest.png](https://github.com/doilonar/Lane-Detection-using-U-Net-with-Spatiotemporal/blob/main/unet_binaryfocal/accuracy_plot_old.png)](https://github.com/doilonar/Lane-Detection-using-U-Net-with-Spatiotemporal/blob/main/unet_iou_loss/accuracy_plot_iou.png)](https://github.com/doilonar/Lane-Detection-using-U-Net-with-Spatiotemporal/blob/main/unet_lstm/accuracy_plot.png))

![Loss LSTM Integration]([[[[https://github.com/tograh/testrepository/3DTest.png](https://github.com/doilonar/Lane-Detection-using-U-Net-with-Spatiotemporal/blob/main/unet_binaryfocal/accuracy_plot_old.png)](https://github.com/doilonar/Lane-Detection-using-U-Net-with-Spatiotemporal/blob/main/unet_binaryfocal/loss_plot_old.png)](https://github.com/doilonar/Lane-Detection-using-U-Net-with-Spatiotemporal/blob/main/unet_iou_loss/loss_plot.png)](https://github.com/doilonar/Lane-Detection-using-U-Net-with-Spatiotemporal/blob/main/unet_lstm/loss_plot.png))
## How to Use

### Prerequisites
You need Python 3 and the following libraries:
-   TensorFlow
-   OpenCV
-   NumPy
-   Matplotlib
-   focal-loss

Install them using pip:
```bash
pip install tensorflow opencv-python numpy matplotlib focal-loss
```

### Training a Model
The training scripts (`run_unet.py`) expect the dataset to be organized into two main directories: one for the input images and one for the ground truth masks.

1.  Place your training images in a folder (e.g., `lane/images/`).
2.  Place the corresponding segmentation masks in another folder (e.g., `lane_detect/masks/`).
3.  Update the `image_folder` and `mask_folder` paths inside the desired `run_unet.py` script.
4.  Execute the script to start training:
    ```bash
    # For U-Net with Binary Focal Loss
    python unet_binaryfocal/run_unet.py

    # For U-Net with ConvLSTM and IoU Loss
    python unet_lstm/run_unet.py
    ```
    The trained model will be saved as `lane_cv_dropout_batch.h5` in the same directory.

### Inference on a Single Image
The `test_image.py` script performs lane detection on a single image and displays the input image, the predicted mask, and a filtered/overlayed image.

1.  Open `test_image.py`.
2.  Set the `model` variable by loading the desired pre-trained model file (e.g., `unet_iou_loss/lane_cv_dropout_batch.h5`).
3.  Update the `image_path` to point to your test image.
4.  Run the script:
    ```bash
    python test_image.py
    ```

### Inference on a Video
The `video_unet.py` script processes a video file frame by frame, overlays the predicted lane segmentation, and displays the result in real-time.

1.  Open `video_unet.py`.
2.  Update `input_video_path` to your video file.
3.  Set the `model` variable by loading the desired pre-trained model. The `unet-lstm` model is recommended for video.
4.  Run the script:
    ```bash
    python video_unet.py
    ```
    Press 'q' to quit the video stream.
![Without LSTM Integration]([[[[https://github.com/tograh/testrepository/3DTest.png](https://github.com/doilonar/Lane-Detection-using-U-Net-with-Spatiotemporal/blob/main/unet_binaryfocal/accuracy_plot_old.png)](https://github.com/doilonar/Lane-Detection-using-U-Net-with-Spatiotemporal/blob/main/unet_iou_loss/accuracy_plot_iou.png)](https://github.com/doilonar/Lane-Detection-using-U-Net-with-Spatiotemporal/blob/main/unet_lstm/accuracy_plot.png)](https://github.com/doilonar/Lane-Detection-using-U-Net-with-Spatiotemporal/blob/main/unet_lstm/whitout_lstm.png))

![With LSTM Integration]([[[[[https://github.com/tograh/testrepository/3DTest.png](https://github.com/doilonar/Lane-Detection-using-U-Net-with-Spatiotemporal/blob/main/unet_binaryfocal/accuracy_plot_old.png)](https://github.com/doilonar/Lane-Detection-using-U-Net-with-Spatiotemporal/blob/main/unet_binaryfocal/loss_plot_old.png)](https://github.com/doilonar/Lane-Detection-using-U-Net-with-Spatiotemporal/blob/main/unet_iou_loss/loss_plot.png)](https://github.com/doilonar/Lane-Detection-using-U-Net-with-Spatiotemporal/blob/main/unet_lstm/loss_plot.png)](https://github.com/doilonar/Lane-Detection-using-U-Net-with-Spatiotemporal/blob/main/unet_lstm/withlstm.png))
