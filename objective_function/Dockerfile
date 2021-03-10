FROM python:3.8

WORKDIR /code

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY data/ data/

COPY objective_server.py objective.py ./

CMD [ "python", "./objective_server.py" ]


