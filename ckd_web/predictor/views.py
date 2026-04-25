from django.shortcuts import render
from .ml_model import predict

def home(request):
    if request.method == 'POST':
        data = request.POST.dict()

        prediction, probability = predict(data)

        result = "CKD Detected" if prediction == 1 else "No CKD"

        return render(request, 'result.html', {
            'result': result,
            'probability': round(probability * 100, 2)
        })

    return render(request, 'home.html')
