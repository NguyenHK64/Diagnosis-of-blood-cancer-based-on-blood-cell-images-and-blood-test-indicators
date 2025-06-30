This project focuses on developing a blood cancer diagnosis model by integrating multimodal data, including blood cell images and complete blood count (CBC) indicators such as WBC, RBC, PLT, age, and gender. This integration allows the model to capture a more comprehensive view of the patient's condition, improving diagnostic accuracy. Blood cell images are preprocessed through color space conversion and segmentation to highlight morphological features, while CBC indicators are normalized to ensure consistency. The model architecture consists of two branches: a lightweight CNN for extracting image features and a neural network for processing numerical data, which are then merged via a fully connected layer.

The model’s performance is evaluated using accuracy, F1-score, and AUC-ROC to ensure effective classification. A prototype diagnostic website is also developed, allowing users to upload images and input test results to receive a preliminary diagnosis. This integrated approach improves diagnostic accuracy (>75%) compared to single-modality models and helps reduce the need for invasive procedures, making it a practical solution for real-world healthcare settings

Link download dataset:

-Blood test cancer: https://drive.google.com/file/d/1n4U5Nf-Ska1b4wjCyWgmt5nIDkUyN7_B/view?usp=drive_link

-Blood cell cancer: https://drive.google.com/file/d/1R7LmIzSv7Nvyb143885soK9IsHYodfll/view?usp=drive_link
