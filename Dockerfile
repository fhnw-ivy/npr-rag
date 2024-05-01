FROM quay.io/jupyter/datascience-notebook:2024-03-14

WORKDIR /usr/src/app
COPY ./requirements.txt ./requirements.txt

RUN conda install -c conda-forge hdbscan
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8888
EXPOSE 6006

CMD ["jupyter", "lab", "--ip='*'", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]