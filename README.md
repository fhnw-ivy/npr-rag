# npr-rag

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
