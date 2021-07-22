FROM python:3.9.2

WORKDIR /KatanAI

RUN apt-get update -y && \
    apt install --no-install-recommends libgl1-mesa-glx -y

RUN apt-get install --no-install-recommends 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python3", "./main.py" ]

# sudo docker build -t katanai .
# docker run katana
# docker run -d ubuntu tail -f /dev/null
