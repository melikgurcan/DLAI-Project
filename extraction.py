from transformers import pipeline

# loading the NER model from Hugging Face
# this model automatically detects Persons (PER), Organizations (ORG), and Locations (LOC)
print("Loading NER Model...")
ner_pipeline = pipeline("ner", aggregation_strategy="simple")

# story context
story = """ Dr. Aris was found dead in the Server Room. 
His assistant, Elara, secretly visited Nexus Corp yesterday. """

# running the model
entities = ner_pipeline(story)

# printing the results
for ent in entities:
    print(f"Extracted Entity: {ent['word']} | Category: {ent['entity_group']} | Confidence Score: {ent['score']:.2f}")