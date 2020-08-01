import boto3
import uuid
from django.conf import settings

class S3utils():
    
    def __init__(self):
        self._session = boto3.session.Session()
        self._client = self._session.client('s3',
            region_name = settings.AWS_S3_REGION_NAME,
            endpoint_url = settings.AWS_S3_ENDPOINT_URL,
            aws_access_key_id = settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key = settings.AWS_SECRET_ACCESS_KEY)

    def getfilename(self):
        return f"smartbins/{str(uuid.uuid4())}.jpg"

    

    def upload(self, filepath):
        filename = self.getfilename()
        #data = pilimage #self.toByteArray(pilimage)
        #print(settings.AWS_STORAGE_BUCKET_NAME, filename)
        #self._client.put_object(ACL='public_read', Bucket=settings.AWS_STORAGE_BUCKET_NAME, Key=filename, Body=data)
        self._client.upload_file(filepath, settings.AWS_STORAGE_BUCKET_NAME, filename, ExtraArgs={'ACL': 'public-read'})
        return f'{settings.AWS_S3_ENDPOINT_URL}/{settings.AWS_STORAGE_BUCKET_NAME}/{filename}'


    def delete(self, filename):
        return self._client.delete_object(Bucket='SIH2020', Key=filename)


S3Connection = S3utils()