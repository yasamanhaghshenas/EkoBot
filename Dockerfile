FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Create db directory ahead of time to avoid permission issues
RUN mkdir -p ./db

EXPOSE 7860

# Use a startup script that handles indexing gracefully
COPY startup.sh .
RUN chmod +x startup.sh

CMD ["./startup.sh"]