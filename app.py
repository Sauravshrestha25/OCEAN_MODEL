from bson import ObjectId
from flask import Flask, redirect, render_template, request, url_for, session
import pandas as pd
import joblib  
from flask_pymongo import PyMongo
from datetime import datetime
import os
import io
import base64
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['MONGO_URI'] = 'mongodb://localhost:27017/ocean_model'
mongo = PyMongo(app)

# Loading the pre-trained model
model = joblib.load('OCEAN_PREDICTION_MODEL_lite.pkl')
# Questions Library
questions = {
    'Extraversion': {
        'EXT1': 'I am the life of the party',
        'EXT2': 'I don’t talk a lot',
        'EXT3': 'I feel comfortable around people',
        'EXT4': 'I keep in the background',
        'EXT5': 'I start conversations', 
        'EXT6': 'I have little to say',
        'EXT7': 'I talk to a lot of different people at parties',
        'EXT8': 'I don’t like to draw attention to myself',
        'EXT9': 'I don’t mind being the center of attention',
        'EXT10': 'I am quiet around strangers'
    },
    'Emotional Stability': {
        'EST1': 'I get stressed out easily',
        'EST2': 'I am relaxed most of the time',
        'EST3': 'I worry about things',
        'EST4': 'I seldom feel blue',
        'EST5': 'I am easily disturbed',
        'EST6': 'I get upset easily',
        'EST7': 'I change my mood a lot',
        'EST8': 'I have frequent mood swings',
        'EST9': 'I get irritated easily',
        'EST10': 'I often feel blue'
    },
    'Agreeableness': {
        'AGR1': 'I feel little concern for others',
        'AGR2': 'I am interested in people',
        'AGR3': 'I insult people',
        'AGR4': 'I sympathize with others\' feelings',
        'AGR5': 'I am not interested in other people\'s problems',
        'AGR6': 'I have a soft heart',
        'AGR7': 'I am not really interested in others',
        'AGR8': 'I take time out for others',
        'AGR9': 'I feel others\' emotions',
        'AGR10': 'I make people feel at ease'
    },
    'Conscientiousness': {
        'CSN1': 'I am always prepared',
        'CSN2': 'I leave my belongings around',
        'CSN3': 'I pay attention to details',
        'CSN4': 'I make a mess of things',
        'CSN5': 'I get chores done right away',
        'CSN6': 'I often forget to put things back in their proper place',
        'CSN7': 'I like order',
        'CSN8': 'I shirk my duties',
        'CSN9': 'I follow a schedule',
        'CSN10': 'I am exacting in my work'
    },
    'Openness': {
        'OPN1': 'I have a rich vocabulary',
        'OPN2': 'I have difficulty understanding abstract ideas',
        'OPN3': 'I have a vivid imagination',
        'OPN4': 'I am not interested in abstract ideas',
        'OPN5': 'I have excellent ideas',
        'OPN6': 'I do not have a good imagination',
        'OPN7': 'I am quick to understand things',
        'OPN8': 'I use difficult words',
        'OPN9': 'I spend time reflecting on things',
        'OPN10': 'I am full of ideas'
    }
}

columns = [
        'EXT1', 'EXT2', 'EXT3', 'EXT4', 'EXT5', 'EXT6', 'EXT7', 'EXT8', 'EXT9', 'EXT10',
        'EST1', 'EST2', 'EST3', 'EST4', 'EST5', 'EST6', 'EST7', 'EST8', 'EST9', 'EST10',
        'AGR1', 'AGR2', 'AGR3', 'AGR4', 'AGR5', 'AGR6', 'AGR7', 'AGR8', 'AGR9', 'AGR10',
        'CSN1', 'CSN2', 'CSN3', 'CSN4', 'CSN5', 'CSN6', 'CSN7', 'CSN8', 'CSN9', 'CSN10',
        'OPN1', 'OPN2', 'OPN3', 'OPN4', 'OPN5', 'OPN6', 'OPN7', 'OPN8', 'OPN9', 'OPN10'
    ]

def predict_personality(questionnaire_data):
    features = [int(questionnaire_data.get(code, 0)) for code in columns]
    
    # prediction
    input_data = pd.DataFrame([features], columns=columns)
    predicted_traits = model.predict(input_data)
    
    trait_names = ['Extraversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
    
    traits_dict = dict(zip(trait_names, predicted_traits[0]))

    return traits_dict

# Home route
@app.route('/')
def index():
    return render_template('index.html', questions=questions)

# CV Form Route
user_data = [] 

@app.route('/submit_cv', methods=['GET', 'POST'])
def submit_cv():
    if request.method == 'POST':
        user_info = {
            'name': request.form.get('name'),
            'email': request.form.get('email'),
            'phone': request.form.get('phone'),
            'address': request.form.get('address'),
            'degree': request.form.getlist('degree[]'), 
            'university': request.form.getlist('university[]'), 
            'grad_year': request.form.getlist('grad_year[]'),  
            'job_title': request.form.getlist('job_title[]'),
            'company': request.form.getlist('company[]'),
            'duration': request.form.getlist('duration[]'),
            'projects': request.form.getlist('projects[]'),
            'project_description': request.form.getlist('project_description[]'),
            'certifications': request.form.getlist('certifications[]'),
            'certification_institution': request.form.getlist('certification_institution[]'),
            'personal_description': request.form.get('personal_description'),
            'skills': request.form.get('skills'),
        }
        
        if not user_info['degree'] or not user_info['university'] or not user_info['grad_year']:
            return "Education information is incomplete. Please fill out all fields.", 400
        
        mongo.db.users.insert_one(user_info)

        session['user_email'] = user_info['email']

        # Redirect to personality questionnaire
        return redirect(url_for('personality_questionnaire'))

    return render_template('user_form.html')

# Personality Questionnaire Route
@app.route('/personality_questionnaire', methods=['GET', 'POST'])
def personality_questionnaire():
    if request.method == 'POST':
        questionnaire_data = {}
        
        for code in request.form:
            questionnaire_data[code] = request.form[code]

        print(questionnaire_data)  
        
        user_email = session.get('user_email') 

        predicted_personality = predict_personality(questionnaire_data)
        mongo.db.users.update_one(
            {'email': user_email},  
            {'$set': {
                'responses': questionnaire_data, 
                'predicted_personality': predicted_personality  
            }}
        )

        return redirect(url_for('thank_you'))

    return render_template('personality_questionnaire.html', questions=questions)


# Thank You Route
@app.route('/thank_you')
def thank_you():
    return render_template('thank_you.html')

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    input_data = {}
    for trait, questions_dict in questions.items():
        for question_id in questions_dict.keys():
            input_data[question_id] = int(request.form[question_id])

    input_df = pd.DataFrame([input_data], columns=columns)

    predicted_traits = model.predict(input_df)
    trait_names = ['Extraversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
    traits_dict = dict(zip(trait_names, predicted_traits[0]))

    user_data[-1]['predicted_personality'] = traits_dict

    user_email = user_data[-1]['cv']['email']  
    mongo.db.users.update_one(
        {'email': user_email}, 
        {'$set': {'predicted_personality': traits_dict}} 
    )

    return render_template('result.html', traits=traits_dict)


#Admin 
@app.route('/admin')
def admin_dashboard():
    users = mongo.db.users.find() 
    user_list = []
    
    for user in users:
        user_info = {
            'name': user.get('name'),
            'email': user.get('email'),
            'degree': user.get('degree'),
            'university': user.get('university'),
            'grad_year': user.get('grad_year'),
            'job_title': user.get('job_title'),
            'company': user.get('company'),
            'duration': user.get('duration'),
            'projects': user.get('projects'),
            'project_description': user.get('project_description'),
            'certifications': user.get('certifications'),
            'certification_institution': user.get('certification_institution'),
            'personal_description': user.get('personal_description'),
            'skills': user.get('skills'),
            'predicted_personality': user.get('predicted_personality'),
            'responses': user.get('responses')
        }
        user_list.append(user_info)

    return render_template('admin_dashboard.html', users=user_list)



@app.route('/user_details/<user_email>')
def user_details_by_email(user_email):
    user = mongo.db.users.find_one({'email': user_email})

    if user:
        predicted_personality = user.get('predicted_personality', {})

        # bar graph for the personality traits
        traits = list(predicted_personality.keys())
        values = [predicted_personality[trait] for trait in traits]

        # Plotting the bar graph
        fig, ax = plt.subplots()
        ax.bar(traits, values, color='skyblue')

        # Adding labels and title
        ax.set_xlabel('Traits')
        ax.set_ylabel('Scores')
        ax.set_title('Predicted Personality Traits')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Save the plot to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='png')
        img_io.seek(0)
        

        img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

        highest_trait = max(predicted_personality, key=predicted_personality.get)
        highest_value = predicted_personality[highest_trait]

        trait_explanations = {
        "Openness": "creativity, curiosity, and openness to new experiences.",
        "Conscientiousness": "self-discipline, organization, and goal-oriented behavior.",
        "Extraversion": "sociability, energy, and a preference for being around others.",
        "Agreeableness": "compassion, cooperation, and trust in others.",
        "Neuroticism": "emotional instability, anxiety, and sensitivity to stress."
        }
        
        if highest_trait in trait_explanations:
            explanation = (
        f"The highest trait is {highest_trait} with a score of {highest_value:.4f}. "
        f"This trait represents the degree to which the individual exhibits {trait_explanations[highest_trait]} "
         )
        else:
             explanation = "Trait information not available."

        return render_template('user_details.html', user=user, graph=img_base64, explanation=explanation)
    else:
        return "User not found", 404




if __name__ == '__main__':
    app.run(debug=True)
