FROM python:3.7.3

WORKDIR ./usr/src/app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

CMD ["/bin/bash"]