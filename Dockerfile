FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch==2.7.1

RUN pip install --no-cache-dir \
    torch-geometric==2.7.0

RUN pip install --no-cache-dir \
    torch-scatter==2.1.2 \
    torch-sparse==0.6.18 \
    torch-cluster==1.6.3 \
    torch-spline-conv==1.2.2 \
    -f https://data.pyg.org/whl/torch-2.7.1+cpu.html

COPY requirements.txt .

#RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r  requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "agent_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]