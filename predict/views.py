from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd

# Create your views here.

def predict(request):
    return render(request, 'predict.html')


def predict_chances(request):

    if request.POST.get('action') == 'post':

        # Reciving Data From Client
        sepal_length = float(request.POST.get('sepal_length'))
        sepal_width = float(request.POST.get('sepal_width'))
        petal_length = float(request.POST.get('petal_length'))
        petal_width = float(request.POST.get('petal_width'))

        # Unpickle model
        model = pd.read.pickle(r"../new_model.pickel")

        # Make Predictions
        result = model.predict([[sepal_width, sepal_length, petal_width, petal_length]])

        classification = request[0]


        return JsonResponse({
                'result': classification,
                'sepal_length': sepal_length,
                'sepal_width': sepal_width,
                'petal_length': petal_length,
                'petal_width': petal_width}, 
                safe=False)