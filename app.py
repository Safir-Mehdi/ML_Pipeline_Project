from flask import Flask, request, jsonify
from src.logger import logging
from src.exception import CustomException
import sys

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
   try:
    #   a = request.form.get('a') if request.method == 'POST' else request.args.get('a')
    #   b = request.form.get('b') if request.method == 'POST' else request.args.get('b')
      c = 4 / 0
      return jsonify(result=c)
   except Exception as e:
       error = CustomException(e, sys)
       logging.info(error.error_message)
       return jsonify({"error": error.error_message}), 400
    #    raise error
   

if __name__ == "__main__":
    app.run(debug=True)
