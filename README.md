# Number Plate Detection and Recognition
This project is a part of the course th Computer Vision at NIT Calicut. The project aims to detect and recognize the number plates of vehicles in an image. The project is divided into two parts:
1. Number Plate Detection
2. Number Plate Recognition


## Installation
1. Clone the repository

    ```bash
    git clone https://github.com/AfthabEK/Number-Plate-Recognition.git
    ```

2. Install the dependencies

    ```bash
    pip install -r requirements.txt
    ```

3. Download the weights for the YOLOv3 model from [here](https://drive.google.com/drive/folders/1XDe29q-N5wm5aq4e9csEtxa8yjV69aTm?usp=drive_linkfolder) and place it in the `./model/weigths` folder.

4. Add the images to be processed in the `./data` folder.



## Usage

1. Add the images to be processed in the data folder.

2. Run the following command to detect the number plates in the images.

    ```bash
    python main.py
    ```

3. To navigate through the plates, click q to quit the current image and move to the next one. 

4.  The detected plates will be saved in result.txt, along with the confidence level.



    