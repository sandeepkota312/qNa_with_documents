import openai
from docx import Document

## Step 1: Mention your OpenAI key 

print("Mention your OpenAI key below(if you don't have one, create a API key in openAI. It will generate the key code)")
openai.api_key = input('Enter your OpenAI API key:')

## Step 2: Use OpenAI to embed multiple documents

doc = Document("demo.docx") # Loading the .docx file(demo.docx)

# My approach is to divide the documents into 3 based on the number of paragraphs
paragraphs = [p.text for p in doc.paragraphs] # Fetching the paragraphs from the document
total_paragraphs = len(paragraphs)
part_size = total_paragraphs // 3   

# Split the paragraphs into three parts
part1 = paragraphs[:part_size]
part2 = paragraphs[part_size:part_size * 2]
part3 = paragraphs[part_size * 2:]

partioned_documents=[part1,part2,part3]

# Generate embeddings for each document part
embeddings = []
for part in partioned_documents:
    response = openai.Completion.create(engine='davinci',  
        prompt=part,
        max_tokens=0,
        temperature=0,
        n=1,
        stop=''
    )
    embeddings.append(response.choices[0].embedding)

## Step 3: Connect to OPENAI API to ask questions and retrieve relevant documents

def retrieve_relevant_document(question):
    # Convert the question to embeddings
    question_response = openai.Completion.create(
        engine='davinci',  # Choose the appropriate openAI model
        prompt=question,
        max_tokens=0,
        temperature=0,
        n=1,
        stop=''
    )
    question_embedding = question_response.choices[0].embedding

    # Find the most relevant document
    best_match_index = 0
    best_match_score = -1

    for i, embedding in enumerate(embeddings):
        similarity_score = openai.TensorflowSimilarityModel.compare_embeddings(
            [question_embedding],
            [embedding])[0][0]
        if similarity_score > best_match_score:
            best_match_score = similarity_score
            best_match_index = i

    # Retrieve the relevant document
    relevant_document = partioned_documents[best_match_index]
    return relevant_document

# Example usage
question=input('Enter your question(example: What does this document tell about?):')
suitable_doc = retrieve_relevant_document(question)

# Use API to answer the question
response = openai.Completion.create(
    engine='davinci',  # Choose the appropriate openAI model
    prompt=question + '\nDocument:' + suitable_doc,
    max_tokens=50,
    temperature=0.7,
    n=1,
    stop=''
)

# Print the answer
print("Question:", question)
print("Relevant Document:", suitable_doc)
answer = response.choices[0].text.strip()
print("Answer:", answer)
