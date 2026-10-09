# Logbook


## Ingest.py

the text from pymupdf.open(pdf_path) contains weird characters "ffi": eﬃciency. Replace? No I added a normalizer from unicodedata.

The source file name field is not really useful but I modified it to have also the page number (more useful)

Ran extra checks to make sure there are no emtpy chunks. 

Maybe change embedding model later to support french?

Changed chunking to split by sentence, more natural. overlap is 2 sentences. 



## retrieval.py

Load FAISS index + chunk metadata, and perform similarity search for a query.

Test with query: natural language question about text (returns whole chunk). Works ~OK: should have the answer in the chunk, and a reasonably low distance. For this model (L2 distance): 
Distance < 0.3–0.5: very strong semantic match (near paraphrase)
Distance 0.5–1.0: good relevant match
Distance 1.0–1.5: loosely related
Distance > 1.5–2.0: likely irrelevant

Works also with french


## generation

There are two options for LLM: local or remote.

-  Local: No need for API key, no limit but uses local resource and can't really be exposed. 

- remote: Create free account on Groq to get an API key. Can be used by a streamlit app. There are safeguards for limits. 

I decided to get and API key for mistral.

The key can be stored in a local file .env that should not be committed (add to .gitignore) and then can be retrieved with dotenv.load_dotenv() and then it's in os.environ{}. Other option is to add the file (somewhere else, not committed) to PATH and then get it. 


# Models

Complete mess. Models mistral-small-latest and mistral-medium-latest give error 429: rate limit exceeded with 0 API requests sent. I ask mistral's AI, tested with curl and found by accident taht mistral-tiny works. I also queried for the list of models (in models.json) and it's not even in it! Claude Sonnet thinks it's an old alias for backward compatibility (but they have aliases...)

Anyway I brute force tested all completion models:

--- SUMMARY ---
Working models (5):
  - codestral-2508
  - ministral-14b-2512
  - ministral-3b-2512
  - ministral-8b-2512
  - voxtral-small-2507

Failing models (3):
  - labs-leanstral-1-5-1: 403
  - mistral-medium-latest: 429
  - mistral-small-2603: 429