# 1. Base image
FROM python:3.9-slim

# 2. Install Streamlit
RUN pip install --no-cache-dir streamlit

# 3. Copy app
WORKDIR /app
COPY app.py .

# 4. Expose port & run
EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
