FROM python:3.6.1

RUN apt-get update && apt-get install -y make autoconf libtool pkg-config

WORKDIR /root
RUN git clone https://github.com/DanBloomberg/leptonica.git
RUN git clone https://github.com/tesseract-ocr/tesseract.git

RUN cd leptonica && ./autogen.sh && ./configure && make && make install
RUN cd tesseract && ./autogen.sh && ./configure && make && make install
RUN ldconfig -v
RUN cd /usr/local/share/tessdata/ && \
        curl -o dan.traineddata https://raw.githubusercontent.com/tesseract-ocr/tessdata/master/dan.traineddata && \
        curl -o eng.traineddata https://raw.githubusercontent.com/tesseract-ocr/tessdata/master/eng.traineddata

COPY requirements.txt requirements.txt
RUN  pip install -r requirements.txt
ADD images images
ADD ocr.py ocr.py

CMD ["python3", "ocr.py"]
