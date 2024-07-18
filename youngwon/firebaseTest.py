from firebase_admin import credentials, initialize_app, storage
# Init firebase with your credentials
cred = credentials.Certificate("./parkjavastorage-firebase-adminsdk-rttma-90b24d3ed3.json")
initialize_app(cred, {'storageBucket': 'parkjavastorage.appspot.com'})

# Put your local file path 
fileName = "test1.jpg" # 저장된 파일 
saveName = "test1" #저장할 이름
bucket = storage.bucket()
blob = bucket.blob(saveName)
blob.upload_from_filename(fileName)

# Opt : if you want to make public access from the URL
blob.make_public()

print("your file url", blob.public_url)

# https://storage.googleapis.com/parkjavastorage.appspot.com/{saveName}