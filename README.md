# AI-Assisted-Early-Dedection-System-for-Ostreoarthritis-OA-
350 character This project develops a low-cost, portable osteoarthritis risk-screening system using MediaPipe gait analysis, ESP32–MPU6050 motion sensors, and questionnaire data. An XGBoost model is planned to classify risk as Low, Moderate, or High. It aims to assist healthcare workers, not replace clinical diagnosis.
# Multimodal AI-Based Early Osteoarthritis Risk Screening System

## 📌 Project Overview
This project aims to develop a low-cost, portable, and non-invasive system to support early osteoarthritis (OA) risk screening. It combines camera-based gait analysis, wearable motion sensors, and questionnaire data to assist healthcare workers, particularly in rural and underserved communities.

The system is designed to work offline and provide preliminary risk indicators. It is intended to support healthcare assessment, not replace professional diagnosis.

## 🎯 Objectives
- Analyze gait and posture using computer vision.
- Collect motion data using wearable sensors.
- Combine movement features with questionnaire responses.
- Develop an AI-based model to categorize risk as Low, Moderate, or High.
- Support accessible screening in primary health centres and rural health camps.

## ⚙️ Technologies Used
- **Python** — data processing and computer vision
- **MediaPipe Pose** — body landmark and movement analysis
- **ESP32** — wearable sensor interface
- **MPU6050** — acceleration and angular-motion sensing
- **Piezoelectric sensors** — planned movement-related vibration sensing
- **XGBoost** — planned machine-learning classification
- **Mobile application** — under development

## 🔧 Hardware Components
- ESP32 development board
- MPU6050 inertial measurement sensor modules
- Piezoelectric sensors
- Jumper wires and connectors
- Knee straps for sensor mounting
- Smartphone camera or computer camera

## 🔄 System Workflow
1. **Patient Information:** Collect relevant information through a questionnaire.
2. **Gait Analysis:** Capture walking video and extract gait and posture features using MediaPipe.
3. **Motion Sensing:** Acquire movement data through the ESP32 and MPU6050 sensor interface.
4. **Feature Integration:** Prepare the questionnaire, camera, and sensor features for combined analysis.
5. **Risk Classification:** Train and integrate an XGBoost model to generate a preliminary risk category.
6. **Report Generation:** Present the screening result and general preventive guidance through the planned application.

## 📊 Current Development Status
- MediaPipe camera processing: Working; measurement accuracy requires improvement.
- ESP32–MPU6050 interface: Tested.
- Mobile application: In progress.
- XGBoost model: Not yet trained.
- Complete system integration and validation: Pending.

## 🚀 Future Improvements
- Improve gait measurement accuracy through calibration and filtering.
- Collect suitable data with appropriate clinical reference labels.
- Train and evaluate the machine-learning model.
- Integrate camera, sensor, and questionnaire features.
- Complete the mobile application and report-generation workflow.
- Evaluate system reliability before practical healthcare use.

## ⚠️ Disclaimer
This project is a research prototype under development. It is not a clinically validated diagnostic system and must not be used as a substitute for professional medical assessment or treatment.

## 👨‍💻 Project Domain
Artificial Intelligence | Computer Vision | Embedded Systems | Wearable Sensors | Healthcare Technology
