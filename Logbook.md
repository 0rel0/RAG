#Logbook


##Ingest.py

the text from pymupdf.open(pdf_path) contains weird characters "ffi": eﬃciency. Replace? No I added a normalizer from unicodedata.

The source file name field is not really useful but I modified it to have also the page number (more useful)

Ran extra checks to make sure there are no emtpy chunks. 