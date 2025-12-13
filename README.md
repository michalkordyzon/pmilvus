````markdown
# 🧠 Milvus Multitenant Demo on IBM Cloud


For streamlit app go [here](https://github.ibm.com/InfiniteLoopLair/milvus/app/README.md)

This project shows how to connect to **IBM Milvus (managed Milvus on IBM Cloud)** from Python and use it as a **vector database for multitenant retrieval**.

You’ll see:

- how to connect to Milvus from code using an API key,
- how to **embed and load documents** into Milvus,
- how to split content into **two collections**:
  - `offerings_public` – visible to everyone,
  - `offerings_managers_only` (or similar) – visible only to managers,
- how to query them so:
  - **all users** get public info,
  - **managers** get access to restricted info as well.

All logic is implemented as a series of Jupyter notebooks.

---

## 🏗 Repository Structure

Top-level files and folders inside `milvus/`:

```
milvus/
├─ 001_load.ipynb
├─ 002_check.ipynb
├─ 003_query.ipynb
├─ 004_metrics.ipynb
├─ 005_roles.ipynb
├─ data/
│  ├─ offerings_public.pdf
│  └─ offerings_managers_only.pdf
├─ example.env
├─ .env   (with real credentials, only on your laptop)
├─ .gitignore
````

### Notebooks

* **`001_load.ipynb`**
  Connects to IBM Milvus and:

  * reads the input PDFs from `data/`,
  * creates Milvus collections (e.g. public + manager-only),
  * computes embeddings and **loads them into Milvus**.

* **`002_check.ipynb`**
  Basic sanity checks:

  * verify that collections exist,
  * check counts, sample documents,
  * ensure the embeddings were inserted correctly.

* **`003_query.ipynb`**
  Simple **semantic search** notebook:

  * connects to Milvus,
  * loads the embedding model,
  * runs queries against the **public collection**,
  * prints the top results (id, score, text, etc.).

* **`004_metrics.ipynb`**
  Experiments with different **similarity metrics** and parameters:

  * `COSINE`, `IP`, `L2` (depending on your index configuration),
  * lets you compare how the results change,
  * useful to debug cases like single-word queries (`Travelflex`) that don’t behave as expected.

* **`005_roles.ipynb`**
  Shows a simple **multitenant / role-based access** pattern:

  * two logical roles: `employee` and `manager`,
  * queries for:

    * **employees** → only public collection,
    * **managers** → manager-only collection,
  * uses the same Milvus instance and embedding model, but different collections.

### Data

* **`data/offerings_public.pdf`**
  Source document for **public** information (e.g. product descriptions available to all users).

* **`data/offerings_managers_only.pdf`**
  Source document for **restricted / manager-only** information
  (e.g. internal rules, pricing guidelines, internal notes).

These PDFs are processed in `001_load.ipynb`, converted into chunks, embedded, and stored as vectors in Milvus.

---

## 🔐 Environment Configuration

The project uses a `.env` file for configuration.
A template is provided in `example.env`:

```env
MILVUS_HOST=XXX
MILVUS_PORT=XXX
MILVUS_API_KEY=XXX
```

### Steps

1. **Copy the template**:

   ```bash
   cp example.env .env
   ```

2. **Edit `.env`** and fill in your IBM Milvus details:

   ```env
   MILVUS_HOST=your-milvus-endpoint-host
   MILVUS_PORT=443
   MILVUS_API_KEY=your-ibm-milvus-api-key
   ```

3. The notebooks will typically read these environment variables and build a Milvus URI like:

   ```python
   import os

   MILVUS_HOST = os.getenv("MILVUS_HOST")
   MILVUS_PORT = os.getenv("MILVUS_PORT")
   MILVUS_API_KEY = os.getenv("MILVUS_API_KEY")
   ```

`.gitignore` is set to ignore `.env`, so your secrets won’t be committed.

---

## 📦 Dependencies

You’ll need a Python environment with (typical) packages like:

* `pymilvus` or `milvus` client for IBM Milvus
* `sentence-transformers`
* `numpy`
* `pandas` (optional, for inspection)
* `python-dotenv` (if used to load `.env` automatically)
* `jupyter` / `notebook` or equivalent

Example installation:

```bash
pip install pymilvus sentence-transformers numpy python-dotenv jupyter
```

*(Adjust to match the exact dependencies you use in your notebooks.)*

---

## 🚀 How to Use the Notebooks

1. **Start Jupyter** in the `milvus/` directory:

   ```bash
   cd milvus
   jupyter notebook
   ```

2. **Open and run notebooks in order:**

   1. `001_load.ipynb`

      * Configure connection
      * Ingest `offerings_public.pdf` into a public collection
      * Ingest `offerings_managers_only.pdf` into a private/manager collection

   2. `002_check.ipynb`

      * Confirm data was loaded correctly

   3. `003_query.ipynb`

      * Test basic semantic search (e.g. query “Travelflex”)
      * Make sure retrieval works on the public collection

   4. `004_metrics.ipynb`

      * Experiment with different similarity metrics and index params
      * Tune retrieval for your use case

   5. `005_roles.ipynb`

      * Play with the **role-based access** demo
      * Try `role = "employee"` vs `role = "manager"` and compare results

---

## 🧩 Multitenant Scenario (Conceptual)

The minimal role system demonstrated:

* **Employee**

  * Only sees data from the **public** collection.
  * Typical queries: customer support, sales, general info.

* **Manager**

  * Can access **restricted internal info**.
  * Same query (e.g. `“Travelflex delays coverage”`) may return:

    * public explanations of the product,
    * plus internal notes / manager-only guidance from the private collection.

IBM Milvus serves both collections from the **same endpoint**, but your application logic (see `005_roles.ipynb`) decides which collection(s) to query based on the user’s role.

---

## 📝 Notes

* Make sure you use the **same embedding model** during ingestion and querying.
* Check that your Milvus index metric (e.g. COSINE/IP/L2) matches the `metric_type` you use in queries.
* For debugging specific terms (like `Travelflex`), combining:

  * vector search (semantic),
  * plus keyword filters (e.g. `text like "%Travelflex%"`)
    can help verify that data is actually present.

---

Made with ❤️ by michal.kordyzon@pl.ibm.com

