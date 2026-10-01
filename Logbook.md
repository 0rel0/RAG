# Logbook


## Ingest.py

the text from pymupdf.open(pdf_path) contains weird characters "ffi": eﬃciency. Replace? No I added a normalizer from unicodedata.

The source file name field is not really useful but I modified it to have also the page number (more useful)

Ran extra checks to make sure there are no emtpy chunks. 

Maybe change embedding model later to support french?



## retrieval.py

Load FAISS index + chunk metadata, and perform similarity search for a query.

Test with query: natural language question about text (returns whole chunk). Works ~OK: should have the answer in the chunk, and a reasonably low distance. For this model (L2 distance): 
Distance < 0.3–0.5: very strong semantic match (near paraphrase)
Distance 0.5–1.0: good relevant match
Distance 1.0–1.5: loosely related
Distance > 1.5–2.0: likely irrelevant

Works also with french