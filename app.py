from flask import Flask, render_template, request, jsonify
from src.pipeline.serving_pipeline import ServingPipeline
from src.utils import create_dataframe
from models.pydantic_model import CensusData
import random

app = Flask(__name__)


def generate_weighted_random_number():
    # Define the weights based on the distribution
    weights = [5670, 26878, 13]
    fnlwgt_digits = [5, 6, 7]
    len = random.choices(fnlwgt_digits, weights=weights, k=1)[0]
    # Generate a random integer with the specified number of digits
    random_int = random.randint(10**(len-1), 10**len - 1)
    return random_int

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the form data from the request
        data = {
            "age": int(request.form.get('age')),
            "fnlwgt": generate_weighted_random_number(),
            "education_num": int(request.form.get('education_num')),
            "capital_gain": int(request.form.get('capital_gain')),
            "capital_loss": int(request.form.get('capital_loss')),
            "hours_per_week": int(request.form.get('hours_per_week')),
            "workclass": request.form.get('workclass'),
            "education": request.form.get('education'),
            "marital_status": request.form.get('marital_status'),
            "occupation": request.form.get('occupation'),
            "relationship": request.form.get('relationship'),
            "race": request.form.get('race'),
            "sex": request.form.get('sex'),
            "native_country": request.form.get('native_country')
        }
        
        # Validate the data using Pydantic model
        try:
            data = CensusData(**data)
        except Exception as e:
            return jsonify({"error": str(e)}), 400
        
        # Create DataFrame from the data object
        X = create_dataframe(data)
        
        # Create serving pipeline instance
        pipeline = ServingPipeline()
        
        pipeline.transform(X)
        prediction = pipeline.predict()
        
        if prediction['pred_income'] == '>50K':
            return render_template('result.html', prediction='You will earn more than 50K$ a year.')
        else:
            return render_template('result.html', prediction='You will earn less than or equal to 50K$ a year.')
    else:
        return render_template('home.html')
    
if __name__ == '__main__':
    app.run(debug=True)