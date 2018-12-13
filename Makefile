file_structure:
	mkdir -p ./images/test
	mkdir -p ./images/errors
	mkdir -p ./images/success
	mkdir -p ./images/test

build: file_structure
	docker build -t ocr .

run: build
	docker run -it -v `pwd`/images:/images ocr

bash: build
	docker run -it ocr /bin/bash



daemon: build
	docker 

build_adb: file_structure
	docker build -t adb_client -f Dockerfile.adb .



run_adb: build_adb 
	docker run -it -v `pwd`/images:/images adb_client 

bash_adb: build_adb
	docker run -it  -v `pwd`/images:/images adb_client /bin/sh

compose_build: file_structure
	docker-compose build 

compose: compose_build
	docker-compose up --force-recreate

down: 
	docker-compose down