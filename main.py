from flask import Flask, request, render_template, jsonify, session
import numpy as np
import pandas as pd
import pickle
from fuzzywuzzy import process

# flask app
app = Flask(__name__)

app.secret_key = 'supersecretkey12345'

# load databasedataset===================================
sym_des = pd.read_csv("Datasets/symtoms_df.csv")
precautions = pd.read_csv("Datasets/precautions_df.csv")
workout = pd.read_csv("Datasets/workout_df.csv")
description = pd.read_csv("Datasets/description.csv")
medications = pd.read_csv('Datasets/medications.csv')
diets = pd.read_csv("Datasets/diets.csv")

# load model===========================================
svc = pickle.load(open('models/svc.pkl', 'rb'))


# ============================================================
# custom and helping functions
# ==========================helper funtions================
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]
    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis]['workout']

    return desc, pre, med, die, wrkout


symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4,
                 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9,
                 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13,
                 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18,
                 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22,
                 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27,
                 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32,
                 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37,
                 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42,
                 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46,
                 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50,
                 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54,
                 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58,
                 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61,
                 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66,
                 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70,
                 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74,
                 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78,
                 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82,
                 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86,
                 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89,
                 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92,
                 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96,
                 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100,
                 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103,
                 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107,
                 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110,
                 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113,
                 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116,
                 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120,
                 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124,
                 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127,
                 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction',
                 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma',
                 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)',
                 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A',
                 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis',
                 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)',
                 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism',
                 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis',
                 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection',
                 35: 'Psoriasis', 27: 'Impetigo'}


# Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]


# creating routes========================================
# list of symptom with there hindi means
symptom_hindi = {
        'itching': 'खुजली',
        'skin_rash': 'चर्म पर चकत्ते',
        'nodal_skin_eruptions': 'त्वचा पर गाँठें',
        'continuous_sneezing': 'लगातार छींक आना',
        'shivering': 'कंपकंपी',
        'chills': 'ठंड लगना',
        'joint_pain': 'जोड़ों में दर्द',
        'stomach_pain': 'पेट दर्द',
        'acidity': 'अम्लता',
        'ulcers_on_tongue': 'जुबान पर घाव',
        'muscle_wasting': 'मांसपेशियों का कमजोर होना',
        'vomiting': 'उल्टी',
        'burning_micturition': 'पेशाब करते समय जलन',
        'spotting_urination': 'पेशाब में खून आना',
        'fatigue': 'थकान',
        'weight_gain': 'वजन बढ़ना',
        'anxiety': 'चिंता',
        'cold_hands_and_feets': 'हाथ और पैर ठंडे होना',
        'mood_swings': 'मनोबल का उतार-चढ़ाव',
        'weight_loss': 'वजन कम होना',
        'restlessness': 'बेचैनी',
        'lethargy': 'ऊर्जा की कमी',
        'patches_in_throat': 'गले में दाग',
        'irregular_sugar_level': 'असामान्य शुगर स्तर',
        'cough': 'खांसी',
        'high_fever': 'बुखार',
        'sunken_eyes': 'अंदर धँसी हुई आँखें',
        'breathlessness': 'साँस फूलना',
        'sweating': 'पसीना आना',
        'dehydration': 'शरीर में पानी की कमी',
        'indigestion': 'पाचन में समस्या',
        'headache': 'सिरदर्द',
        'yellowish_skin': 'पीली त्वचा',
        'dark_urine': 'गहरे रंग का पेशाब',
        'nausea': 'उल्टी जैसा महसूस होना',
        'loss_of_appetite': 'भूख का कम होना',
        'pain_behind_the_eyes': 'आँखों के पीछे दर्द',
        'back_pain': 'पीठ दर्द',
        'constipation': 'कब्ज',
        'abdominal_pain': 'पेट में दर्द',
        'diarrhoea': 'दस्त',
        'mild_fever': 'हल्का बुखार',
        'yellow_urine': 'पीला पेशाब',
        'yellowing_of_eyes': 'आँखों का पीला होना',
        'acute_liver_failure': 'तीव्र यकृत विफलता',
        'fluid_overload': 'शरीर में तरल का अधिक होना',
        'swelling_of_stomach': 'पेट का फूलना',
        'swelled_lymph_nodes': 'सूजे हुए लसीका ग्रंथि',
        'malaise': 'बेचैनी का अनुभव',
        'blurred_and_distorted_vision': 'धुंधला और विकृत दृश्य',
        'phlegm': 'कफ',
        'throat_irritation': 'गले में जलन',
        'redness_of_eyes': 'आँखों का लाल होना',
        'sinus_pressure': 'साइनस का दबाव',
        'runny_nose': 'नाक बहना',
        'congestion': 'नाक का बंद होना',
        'chest_pain': 'छाती में दर्द',
        'weakness_in_limbs': 'हाथ-पैरों में कमजोरी',
        'fast_heart_rate': 'दिल की धड़कन तेज होना',
        'pain_during_bowel_movements': 'पेट साफ करते समय दर्द',
        'pain_in_anal_region': 'गुदा क्षेत्र में दर्द',
        'bloody_stool': 'खून वाला मल',
        'irritation_in_anus': 'गुदा में जलन',
        'neck_pain': 'गर्दन में दर्द',
        'dizziness': 'चक्कर आना',
        'cramps': 'मांसपेशियों में ऐंठन',
        'bruising': 'चोट के निशान',
        'obesity': 'मोटापा',
        'swollen_legs': 'सूजी हुई टाँगें',
        'swollen_blood_vessels': 'सूजी हुई रक्त वाहिकाएँ',
        'puffy_face_and_eyes': 'मुड़े हुए चेहरे और आँखें',
        'enlarged_thyroid': 'वृद्धित थायरॉइड',
        'brittle_nails': 'नाजुक नाखून',
        'swollen_extremeties': 'सूजे हुए अंग',
        'excessive_hunger': 'अत्यधिक भूख',
        'extra_marital_contacts': 'वैवाहिक बाहर संपर्क',
        'drying_and_tingling_lips': 'सूखते और झुनझुनाती हुई होंठ',
        'slurred_speech': 'गड़बड़ बोलना',
        'knee_pain': 'घुटने में दर्द',
        'hip_joint_pain': 'कूल्हे के जोड़ में दर्द',
        'muscle_weakness': 'मांसपेशियों में कमजोरी',
        'stiff_neck': 'गर्दन में अकड़न',
        'swelling_joints': 'सूजे हुए जोड़',
        'movement_stiffness': 'आंदोलन में कठोरता',
        'spinning_movements': 'घूमते हुए आंदोलन',
        'loss_of_balance': 'संतुलन खोना',
        'unsteadiness': 'अस्थिरता',
        'weakness_of_one_body_side': 'शरीर के एक हिस्से में कमजोरी',
        'loss_of_smell': 'गंध की हानि',
        'bladder_discomfort': 'मूत्राशय में असुविधा',
        'foul_smell_of_urine': 'पेशाब की बदबू',
        'continuous_feel_of_urine': 'पेशाब का लगातार महसूस होना',
        'passage_of_gases': 'गैसों का निकलना',
        'internal_itching': 'अंदर की खुजली',
        'toxic_look_(typhos)': 'जहरीला लुक (टायफॉस)',
        'depression': 'अवसाद',
        'irritability': 'चिड़चिड़ापन',
        'muscle_pain': 'मांसपेशियों में दर्द',
        'altered_sensorium': 'संवेदनाओं में बदलाव',
        'red_spots_over_body': 'शरीर पर लाल धब्बे',
        'belly_pain': 'पेट दर्द',
        'abnormal_menstruation': 'अनियमित माहवारी',
        'dischromic_patches': 'वर्णहिन धब्बे',
        'watering_from_eyes': 'आँखों से पानी बहना',
        'increased_appetite': 'भूख का बढ़ना',
        'polyuria': 'अत्यधिक मूत्र उत्पादन',
        'family_history': 'परिवार का इतिहास',
        'mucoid_sputum': 'सारयुक्त बलगम',
        'rusty_sputum': 'जंग जैसा बलगम',
        'lack_of_concentration': 'एकाग्रता की कमी',
        'visual_disturbances': 'दृष्टि में गड़बड़ी',
        'receiving_blood_transfusion': 'रक्त ट्रांसफ्यूजन लेना',
        'receiving_unsterile_injections': 'असुरक्षित इंजेक्शन्स लगवाना',
        'coma': 'कोमा',
        'stomach_bleeding': 'पेट से खून आना',
        'distention_of_abdomen': 'पेट का फूलना',
        'history_of_alcohol_consumption': 'शराब पीने का इतिहास',
        'fluid_overload': 'तरल का अधिक होना',
        'blood_in_sputum': 'बलगम में खून',
        'prominent_veins_on_calf': 'पिंडली पर उभरी हुई नसें',
        'palpitations': 'दिल की धड़कन तेज होना',
        'painful_walking': 'चलते समय दर्द',
        'pus_filled_pimples': 'पुसी हुई पिंपल्स',
        'blackheads': 'ब्लैकहेड्स',
        'scurring': 'सक्रियता',
        'skin_peeling': 'त्वचा का छिलना',
        'silver_like_dusting': 'चांदी जैसा पाउडर',
        'small_dents_in_nails': 'नाखूनों में छोटे-छोटे डेंट्स',
        'inflammatory_nails': 'प्रदाहनशील नाखून',
        'blister': 'फफोला',
        'red_sore_around_nose': 'नाक के आसपास लाल घाव',
        'yellow_crust_ooze': 'पीली परत वाली स्राव',
        'prognosis': 'रोग का पूर्व'
    }

@app.route("/")
def index():
    return render_template("index.html",symptom_hindi=symptom_hindi)


# Define a route for the home page
@app.route('/predict', methods=['GET', 'POST'])
@app.route('/predict', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms').lower()

        # If the input is empty
        if not symptoms:
            session['message'] = "Please enter symptoms correctly."
            return render_template('index.html', message=session['message'],symptom_hindi=symptom_hindi)

        user_symptoms = [s.strip() for s in symptoms.split(',')]
        valid_symptoms = []
        score = 0
        for symptom in user_symptoms:
            score = 0
            best_match, score = process.extractOne(symptom, symptoms_dict.keys())
            if score >= 80:  # Adjust the threshold as needed
                valid_symptoms.append(best_match)

        # If there are no valid symptoms
        if score < 79:
            session['message'] = "Please either write symptoms correctly or you have written misspelled symptoms."
            return render_template('index.html', message=session['message'],symptom_hindi=symptom_hindi)

        # Otherwise, proceed with prediction
        user_symptoms = [symptom.strip("[]' ") for symptom in valid_symptoms]
        predicted_disease = get_predicted_value(user_symptoms)
        dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)

        my_precautions = [i for i in precautions[0]]

        # Clear the message after the next request
        session.pop('message', None)  # Remove the message after successful processing

        return render_template('index.html', predicted_disease=predicted_disease, dis_des=dis_des,
                               my_precautions=my_precautions, medications=medications, my_diet=rec_diet,
                               workout=workout , symptom_hindi=symptom_hindi)
    return render_template('index.html', symptom_hindi=symptom_hindi)


# about view funtion and path
@app.route('/about')
def about():
    return render_template("about.html")


# contact view funtion and path
@app.route('/contact')
def contact():
    return render_template("contact.html")


# developer view funtion and path
@app.route('/developer')
def developer():
    return render_template("developer.html")


# about view function and path
@app.route('/blog')
def blog():
    return render_template("blog.html")


if __name__ == '__main__':
    app.run(debug=True)
