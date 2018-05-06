from flask import Flask, request
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

class PredictSimple(Resource):
    def get(self, input):
        tokenizer.fit_on_texts(input)
        prediction = model.predict(pad_sequences(tokenizer.texts_to_sequences(input)))
        return {"svar": prediction}
        
api.add_resource(PredictSimple, '/<string:input>')

if __name__ == '__main__':
    app.run(debug=True)
