from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_path = "./modernBERT"

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForSequenceClassification.from_pretrained(model_path)

conversation = """
Summary:
Name of the user is Rishi. The user is building a second-brain application with a retrieval pipeline.
The system stores knowledge summaries and detailed knowledge and retrieves
them when they are relevant to a conversation.

Recent Messages:

User: I'm trying to decide how much previous context I should give the
retrieval classifier.
Retrieved knowledge:
Previous experiments showed that providing a conversation summary together
with recent messages gives the model enough information to determine whether
additional knowledge is useful.
Rishi recently completed and passed the language common test exam, demonstrating proficiency in english and hindi and other languages.

Assistant: You should include the recent messages along with any knowledge
that was retrieved for those messages, because that represents the context
the user has already been exposed to.
Retrieved knowledge:
The retrieval pipeline should avoid returning information that is already
present in the conversation context.
Rishi recently completed and passed the language common test exam, demonstrating proficiency in english and hindi and other languages.

User: So the classifier needs to understand both what the user has already
seen and what the new candidate contains.
Retrieved knowledge:
A retrieval decision can be formulated as a classification problem where
the model receives the current conversational context and a candidate
knowledge chunk.
Rishi recently completed and passed the language common test exam, demonstrating proficiency in english and hindi and other languages.

Query:
If a candidate contains information that the conversation hasn't seen yet
but is highly relevant to the current question, should the classifier mark
it as needed?
""".strip()


context = """
Rishi recently completed and passed the language common test exam, demonstrating proficiency in english and hindi and other languages.
""".strip()

inputs = tokenizer(
    conversation,
    text_pair = context,
    truncation = True,
    return_tensors = "pt"
)

model.eval()

with torch.no_grad():
    outputs = model(**inputs)
    print(outputs)