# npr Mini Challenge 1: Retrieval Augmented Generation (RAG)

## Project Artifacts
The following section shows an overview of the project's artifacts including the thoughts and "raison d'être" behind every component.

### Notebooks
The following list shows the relevant notebooks of this project. Every Jupyter Notebook also was exported into an `.html` file so the outputs can be viewed without running a Jupyter Server. These exports can be found in the `/exports` folder.
- **Main Notebook**: The main notebook (`main.ipynb` in the Project's root) was used to explore different ways to build a RAG pipeline. It's the main artifact of this project and holds the team's observations and conclusions on the results the baseline system and its extensions yielded. The notebook guides you through all explorations, starting from monitoring, through preprocessing, chunking and embedding to all experiments. **DISCLAIMER**: If you plan to run the `main.ipynb` please consult the **Setup section** inside the Main-Notebook that will give you more information on available runtime settings (i.e. for Caching and Monitoring).
- **Exploration Notebook**: The exploration notebook (in `/notebooks/exploration.ipynb`) holds the exploratory data analysis of the challenge's dataset ([Cleantech Media Dataset](https://www.kaggle.com/datasets/jannalipenkova/cleantech-media-dataset)). The observations in that notebook led to a lot of initiatives inside the Preprocessing step of the project found in `/src/preprocessing.py`. 
- **MVP Notebook**: The mvp notebook (in `/notebooks/mvp.ipynb`) holds the teams first fully working RAG pipeline which then was consolidated in the `main.ipynb` notebook.
- **Eval Mapping**: To evaluate each explored RAG system it was necessary to map each evaluation sample to its relevant chunk. This entire process is described inside the `/notebooks/eval_mapping.ipynb`.

### Scripts
The scripts used in this challenge are listed here.
- **Subset Generation**: In order to develop in a lightweight environment the `generate_subset.py` script helps to reduce the size of the dataset to a number of samples `n`. This way the vector store doesn't need to get ingested with the full size of the dataset and saves some time if the embedding and retrieval step gets changed.
- **Testset Generation**: Additionally to the subset generation there is a testset generator (in `/scripts/generate_testset.py`) that acts as a wrapper around the RAGAS `TestsetGenerator`. This way we are able to control how we would like the evaluation set to be distributed.

### Codebase (`src`)
There are numerous processes under the hood that are used in the `main.ipynb`. These components are listed here:
- **Evaluation**: The `/src/evaluation.py` holds the main component for evaluation, namely the `Evaluator`. This class steers the entire evaluation process for all explored experiments inside `main.ipynb`. This abstraction allowed for a common place for all explored systems to get rated with the same metrics and same adaptations of the evaluation set. Since we mainly used RAGAS for evaluation we also had to wrap our evaluation set into a RAGAS-digestible format. To achieve that, the `evaluation.py` also holds a `DatasetCreator` that aligns our evaluation set with the RAGAS structure.
- **Generation**: The `/src/generation.py` functionality streamlines the LLM that is used inside the `main.ipynb` notebook. The `get_llm_model` function allows us to control the model that should and its temperature in one place.
- **Preprocessing**: The `Preprocessor` class inside `/src/preprocessing.py` allows for a controlled and streamlined way of removing unwanted noise in the Cleantech Dataset. The preprocessing measures taken mainly stem from observations made inside the `exploration.ipynb` notebook. This component removes duplicate chunks, non-english language chunks and chunks with special characters that wouldn't add meaningful information.
- **Vector Store**: The `VectorStore` class inside `/src/vector_store.py` wraps around the `Chroma` object, allowing for a structured way to control how the Vector Store can be accessed.

### Use of AI
The excerpt on the usage of assistants like ChatGPT and GitHub CoPilot was written in [`USE-OF-AI.md`](USE-OF-AI.md) in the root folder of the project.

## Download ChromaDB SQLite database

Download the ChromaDB SQLite database from the following
link: [ChromaDB](https://fhnw365-my.sharepoint.com/:f:/g/personal/noah_leuenberger_students_fhnw_ch/EhYOpVb2VzRMpr6nHtanNrgBychAJzcV7HsjMHfaYAbMGQ?e=V2nYRz).

After downloading the database, place it in the root directory of the project. The expected path structure should look
like this:

```
npr-rag/
    chroma/
        chroma.sqlite3
    ...
...
```

## Docker Setup

### Requirements

- **Docker**: Install from [here]([https://hub.docker.com/](https://www.docker.com/products/docker-desktop/)).
- **Docker Compose**: Install from [Docker Compose Installation Guide](https://docs.docker.com/compose/install/).

### Using Docker Compose (Recommended)

1. **Start the JupyterLab server**:
    ```bash
    docker-compose up
    ```
   Access the server at `http://localhost:8888`. The project directory is mounted within the container for real-time
   file synchronization.

### Alternative Method: Using Dockerfile Directly

1. **Build the Docker image**:
    ```bash
    docker build -t npr-rag-jupyterlab .
    ```

2. **Run the Docker container**:
    ```bash
    docker run -p 8888:8888 -v "$(pwd):/usr/src/app" npr-rag-jupyterlab
    ```
   Navigate to `http://localhost:8888` in your web browser.
