# CardioVision | Cardiac Arrest Prediction using ECG

A deep learning-based diagnostic application for analyzing ECG signals and predicting cardiac arrest. This project processes 1K+ ECG images, achieving 85-90% accuracy in detecting cardiovascular anomalies.

## Overview

CardioVision is a comprehensive machine learning system that analyzes ECG (Electrocardiogram) signals to predict cardiac arrest and other cardiovascular diseases. The system uses TensorFlow for deep learning model training and Streamlit for the user interface, with deployment capabilities on AWS Lambda for real-time inference.

## Features

- **ECG Signal Analysis**: Processes 1K+ ECG images with advanced preprocessing
- **Deep Learning Models**: TensorFlow-based models for accurate prediction
- **Real-time Inference**: Deployed on AWS Lambda for low-latency predictions
- **Interactive Dashboard**: Streamlit-based interface for easy interaction
- **Automated Reports**: Generates clinical anomaly summaries for healthcare professionals
- **High Accuracy**: Achieves 85-90% accuracy in cardiac arrest prediction

## Technologies Used

- **TensorFlow**: Deep learning framework for model training
- **Streamlit**: Web application framework for interactive dashboards
- **AWS Lambda**: Serverless deployment for real-time inference
- **Flask**: Backend framework for API services
- **Python**: Core programming language
- **NumPy/Pandas**: Data processing and manipulation
- **OpenCV**: Image preprocessing for ECG signals

## Project Structure

```
CardioVision/
├── Cardiovascular-Detection-using-ECG-images-main/
│   ├── colabs/                    # Jupyter notebooks for analysis
│   ├── Combined1d_csv/            # Combined 1D ECG data
│   ├── Deployment/                 # Production deployment code
│   ├── docs/                      # Project documentation
│   ├── ECG_IMAGES_DATASET/        # ECG image dataset
│   ├── Final_Dataset/             # Processed datasets
│   ├── model_pkl/                 # Trained model files
│   ├── preproceed_images/          # Preprocessed images
│   └── preprocessed_1d/            # Preprocessed 1D signals
├── Scaled_1DLead_*.csv            # Scaled ECG lead data
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/aryamanjalali/CardioVision.git
cd CardioVision
```

2. Install dependencies:
```bash
pip install tensorflow streamlit flask numpy pandas opencv-python
```

3. For deployment, install AWS Lambda dependencies:
```bash
pip install -r Deployment/requirements.txt
```

## Usage

### Local Development

1. Run the Streamlit application:
```bash
cd Cardiovascular-Detection-using-ECG-images-main/Deployment
streamlit run final_app.py
```

2. Access the dashboard at `http://localhost:8501`

### Model Training

1. Open the Jupyter notebooks in `colabs/` directory
2. Follow the training pipeline in the notebooks
3. Save trained models to `model_pkl/` directory

### AWS Lambda Deployment

1. Package the deployment code:
```bash
cd Deployment
zip -r lambda_function.zip .
```

2. Upload to AWS Lambda and configure the handler

## Results

- **Accuracy**: 85-90% in cardiac arrest prediction
- **Dataset**: 1K+ ECG images processed
- **Deployment**: Real-time inference on AWS Lambda
- **Latency**: Low-latency predictions for clinical use

## Author

**Aryaman Jalali**
- GitHub: [@aryamanjalali](https://github.com/aryamanjalali)
- LinkedIn: [aryamanjalali](https://www.linkedin.com/in/aryamanjalali/)
- Email: aryamanj@bu.edu

## License

This project is for educational and research purposes.

## Acknowledgments

- ECG dataset providers
- TensorFlow and Streamlit communities
- Healthcare professionals for domain expertise
