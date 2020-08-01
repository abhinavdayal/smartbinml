from rest_framework.views import APIView
from rest_framework.response import Response
from PIL import Image
from io import BytesIO
from . import classifier
from base64 import b64decode

# Create your views here.

class CaffeDetector(APIView):
    def post(self, request):
        try:
            image_data = b64decode(request.data['data'])
            width = request.data['width']
            height = request.data['height']
            mode = request.data['mode']
            
            sample = Image.open(BytesIO(image_data))
            response = classifier.getPredictionsFor([sample], net, mean)
            return Response(response, status=201)
        except Exception as e:
            return Response(e.__str__(), status.HTTP_400_BAD_REQUEST)