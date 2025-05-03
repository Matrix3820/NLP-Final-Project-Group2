# **DATS 6312: NLP Final Project - 02 May 2025**
## RAGnRecipe - A Medically Aware Recipe Generator

### Directory Structure
```
├── Code
│   ├── Preprocessing
│   │   ├── EDA.py
│   │   └── preprocess_csv.py
│   ├── RAG
│   │   ├── .env
│   │   ├── generate_rag_data_format.py
│   │   ├── generate_rag_ingredients.py
│   │   ├── llm.py
│   │   └── sample_upsert.py
│   ├── app.py
│   └── requirements.txt
├── Data
│   ├── Rag_Ingredients.jsonl
│   ├── Rag_Recipes.jsonl
│   ├── recipes_data.csv
```

<br><br>
### Datasets
Due to copyright and size restrictions, the dataset can not be uploaded in this repository.
1. Recipe Dataset - [Download](https://www.kaggle.com/datasets/wilmerarltstrmberg/recipe-dataset-over-2m)

**Download and Save This in the Data folder as recipes_data.csv**

**Note** : You will have to create a folder named **Data** in the **Project Root**. You can follow the Directory structure above for reference

<br><br>
#### Additionl Datafiles Created:
These files will be needed only if you want to Create the app on your machine with the Same subset of Data use and the Same Ingredient Level Medical Facts by uploading them to Pinecone. You can create your own Sample of recipes with different labels(if needed) using the codes in this repository. Download [here](https://drive.google.com/drive/folders/1RGv4HRkvi0mqTIc2KmO8iJFov362eHCa?usp=sharing)

1. Rag_Recipes.jsonl
2. Rag_Ingredients.jsonl
3. recipes_data_sampled_labelled.csv

You can also use wget to fetch this data using wget through the instructions below:

```shell
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ZhXBIm-P5cl0HMbNJldzb1Ycgrxq-Aqq' -O data.zip
unzip data.zip
```
Once Done, Move the files to the Data Folder

<br><br>
### Install additional requirements:
In addition to the pre installed packages provided on our server a few need to be installed
```shell
pip install -r Code/requirements.txt
```
**Note** : The entire list of packages used on the server will be update here at a later time

<br><br>
### Instructions to run Streamlit App
If you want to view the already created App you can run the below command -- **Note** : This will **not work after 17th May 2025** - You will need to rebuild the app with your Pinecone/HuggingFaceHub keys and tokens with the commands in the below sections
```shell
streamlit run Code/app.py --server.port
```
**Note**: Sometimes, Due to Memory issues or multiple calls, the app crashes - Kill the process to free up memory - Re run the Streamlit app and try the query again - If Issue still persists you can view the output using the below command

```shell
python3 Code/RAG/llm.py
```
You will be prompted to enter your query

<br><br>
### Instructions to run all codes - Build system from scratch
These Instructions are if you want to Modify the sample data and Store Data in your own vector Database - The code currently Works off of my Access Tokens and All Tokens will be deleted after the Date mentioned at the start of this document for security purposes.
**The below Steps will be necessary after  17th May 2025.**

After downloading the repository and the Recipe Dataset open the terminal and be in the Project ROOT Folder and run the below commands in order: 
  
```shell
python3 Code/Preprocessing/preprocess_csv.py
```

```shell
python3 Code/RAG/generate_rag_data_format.py
```

```shell
python3 Code/RAG/generate_rag_ingredients.py
```

Before the next Command, Go ahead and Modify the Code/Rag/.env file - Update it with your API Key and Environment

```
PINECONE_API_KEY=[your API Key]
PINECONE_ENVIRONMENT=[your Environment Region]
RECIPE_INDEX_NAME=[name of recipe index]
FACTS_INDEX_NAME=[name of facts index]
EMBEDDING_MODEL_NAME=[embedding model you want to use
DIMENSION=[dimension of that embedding model
```
**Note**: EMBEDDING_MODEL_NAME and DIMENSION are related to each other. Make sure both get updated if you change either one. Some Code changes will be required in sample_upsert.py and llm.py if you use a non sentence transformer model

```shell
python3 Code/RAG/sample_upsert.py
```

For the next command you will need Model Access from HuggingFace which can be requested [here](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1). Create you own API Key and **modify line number 12 in llm.py**
```shell
python3 Code/RAG/llm.py
```
You will be prompted to enter your query


