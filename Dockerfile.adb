FROM tabrindle/min-alpine-android-sdk:latest

RUN apk update && apk add python3 gcc python3-dev
COPY requirements_adb.txt requirements.txt


RUN apk add tesseract-ocr jpeg-dev zlib-dev build-base linux-headers 
RUN pip3 install -r requirements.txt
COPY android.py .

CMD ["python3", "android.py"]
