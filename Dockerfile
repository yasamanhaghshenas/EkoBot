FROM python:3.11-slim

WORKDIR /app


COPY . .


RUN pip install --no-cache-dir -r requirements.txt


RUN mkdir -p ./db


EXPOSE 7860


RUN chmod +x startup.sh


CMD ["./startup.sh"]
