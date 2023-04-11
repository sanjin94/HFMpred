FROM python:slim

RUN apt-get update && apt-get upgrade -y

RUN apt-get install gcc g++ musl-dev -y

RUN python -m pip install --upgrade pip

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "HFMpred_app.py", "--server.port=8501", "--server.address=0.0.0.0"]