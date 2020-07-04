FROM python:3.7.3

ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py --user && \
	rm get-pip.py

RUN pip install --user --upgrade pip
RUN pip install --user numpy matplotlib pillow pandas requests
RUN pip install --user opencv-python
RUN pip install --user hydra-core --upgrade
RUN pip install --user omegaconf
RUN pip install --user mypy

CMD ["/bin/bash"]