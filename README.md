# Medicine Recommendation System (AI Doctor)

## Overview
The **Medicine Recommendation System** is an AI-based application designed to recommend medicines based on symptoms provided by the user. The system uses advanced machine learning algorithms to analyze the input symptoms and suggest suitable medications. It aims to assist users in identifying potential treatment options for various health conditions.

## Features
- **Symptom Input:** Users can enter a list of symptoms they are experiencing.
- **Medicine Recommendation:** Based on the entered symptoms, the system provides a list of potential medicines.
- **User-Friendly Interface:** A clean and easy-to-use interface that allows for smooth interaction with the system.
- **Real-Time Analysis:** The system analyzes symptoms and suggests medicine in real-time.

## Tech Stack
- **Frontend:** 
  - HTML
  - CSS (for styling and layout)
  - JavaScript (for interactivity)
- **Backend:**
  - Python (Flask framework for backend API)
  - Scikit-learn (for machine learning and recommendation logic)
- **Database:**
  - (Optional) A simple JSON or SQLite database to store symptoms and medicine mappings

## Installation

### Prerequisites
- Python 3.x
- Git
- A modern web browser (e.g., Chrome, Firefox)

### Steps to run the project locally:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/medicine-recommendation-system.git
   
2. **Navigate to the project folder:**
   ```bash
   cd medicine-recommendation-system
3. **Install dependencies: Make sure you have all required Python packages installed. You can install them using:**
   ```bash
   pip install -r requirements.txt
4. **Run the flask app**
   ```bash
   python app.py
## How to Use

1. Launch the application in your web browser.
2. On the homepage, you will find an input form where you can enter the symptoms you are experiencing.
3. After entering the symptoms, click on the **Predict** button.
4. The system will display a list of recommended medicines based on the symptoms you provided.

## Example Input and Output

### Input:
Symptoms: **"Fever, Cough, Headache"**

### Output:
Recommended Medicines:
- Paracetamol
- Ibuprofen
- Cough Syrup

## Future Enhancements
- **Additional Symptom Analysis:** Improve the symptom-to-medicine mapping by using more advanced algorithms.
- **Personalization:** Implement a user profile to recommend medicines based on the user's previous history.
- **Integration with Health APIs:** Integrate with healthcare APIs to provide more accurate and real-time data.

## Contributing
Feel free to fork this repository, make changes, and submit pull requests. Contributions are welcome!
