FROM python:3.11-slim

WORKDIR /srv/iaq

# Crear usuario sin privilegios de root (mitigación STRIDE Elevation of Privilege)
RUN useradd -m -u 1000 iaq_user

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
RUN chown -R iaq_user:iaq_user /srv/iaq

USER iaq_user
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]