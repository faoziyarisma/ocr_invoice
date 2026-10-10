FROM python:3.12-slim

WORKDIR /eval
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY run_eval.py metrics.py compare_versions.py ./

ENTRYPOINT ["python", "run_eval.py"]
