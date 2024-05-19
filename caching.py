import json
import time

import chromadb
import faiss
from datasets import load_dataset
from loguru import logger
from sentence_transformers import SentenceTransformer


logger.info("loading the dataset")
data = load_dataset("keivalya/MedQuad-MedicalQnADataset", split='train')

data = data.to_pandas()
data["id"]=data.index

subset_data = data.head(100)
logger.info(subset_data.sample(5))

DOCUMENT="Answer"
TOPIC="qtype"


chroma_client = chromadb.PersistentClient(path = "vector_db/")
collection_name = "meddb"

if len(chroma_client.list_collections()) > 0 and collection_name in [chroma_client.list_collections()[0].name]:
  logger.info(f"Collection {collection_name} already exists")
  collection = chroma_client.get_collection(name = collection_name)
  pass
else:
  logger.info(f"Creating the collection {collection_name}")
  collection = chroma_client.create_collection(name = collection_name)
  collection.add(
      documents=subset_data[DOCUMENT].tolist(),
      metadatas=[{TOPIC: topic} for topic in subset_data[TOPIC].tolist()],
      ids=[f"id{x}" for x in range(100)],
  )

def query_database(query_text, n_results=10):
    results = collection.query(query_texts=query_text, n_results=n_results)
    return results

def init_cache():
  index = faiss.IndexFlatL2(768)

  encoder = SentenceTransformer('all-mpnet-base-v2')
  return index, encoder

def retrieve_cache(json_file):
  try:
    with open(json_file, 'r') as file:
      cache = json.load(file)
  except FileNotFoundError:
      cache = {'questions': [], 'embeddings': [], 'answers': [], 'response_text': []}

  return cache

def store_cache(json_file, cache):
  with open(json_file, 'w') as file:
    json.dump(cache, file)


class semantic_cache:
  def __init__(self, json_file="cache_file.json", threshold=0.35):
      # Initialize Faiss index with Euclidean distance
      self.index, self.encoder = init_cache()

      # Set Euclidean distance threshold
      # a distance of 0 means identicals sentences
      # We only return from cache sentences under this thresold
      self.euclidean_threshold = threshold

      self.json_file = json_file
      self.cache = retrieve_cache(self.json_file)

  def ask(self, question: str) -> str:
      # Method to retrieve an answer from the cache or generate a new one
      start_time = time.time()
      try:
          #First we obtain the embeddings corresponding to the user question
          embedding = self.encoder.encode([question])

          # Search for the nearest neighbor in the index
          self.index.nprobe = 8
          D, I = self.index.search(embedding, 1) #returns the distance to the nearest vector and the indices

          if D[0] >= 0:
              if I[0][0] >= 0 and D[0][0] <= self.euclidean_threshold:
                  row_id = int(I[0][0])

                  logger.info('Answer recovered from Cache. ')
                  logger.info(f'{D[0][0]:.3f} smaller than {self.euclidean_threshold}')
                  logger.info(f'Found cache in row: {row_id} with score {D[0][0]:.3f}')
                  logger.info('response_text: ' + self.cache['response_text'][row_id])

                  end_time = time.time()
                  elapsed_time = end_time - start_time
                  logger.info(f"Time taken: {elapsed_time:.3f} seconds")
                  return self.cache['response_text'][row_id]

          answer  = query_database([question], 1)
          response_text = answer['documents'][0][0]

          self.cache['questions'].append(question)
          self.cache['embeddings'].append(embedding[0].tolist())
          self.cache['answers'].append(answer)
          self.cache['response_text'].append(response_text)

          logger.info('Answer recovered from ChromaDB. ')
          logger.info(f'response_text: {response_text}')

          self.index.add(embedding)
          store_cache(self.json_file, self.cache)
          end_time = time.time()
          elapsed_time = end_time - start_time
          logger.info(f"Time taken: {elapsed_time:.3f} seconds")
          logger.info('\n\n\n\n\n')

          return response_text
      except Exception as e:
          raise RuntimeError(f"Error during 'ask' method: {e}")

cache = semantic_cache('medcache.json')

results = cache.ask("what vaccine for La Crosse encephalitis virus (LACV)?")

results = cache.ask("list down vaccines and precautions against La Crosse encephalitis virus (LACV)?")

