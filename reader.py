import pandas as pd
import os
import torch
from transformers import RobertaTokenizer, RobertaForQuestionAnswering
from transformers.models.transfo_xl.tokenization_transfo_xl import tokenize_numbers
from transformers import pipeline

class roberta:

    def __init__(self,docs,query) -> None:
        self.retrived_ans = docs
        self.question = query 
        self.current_dir = os.getcwd()
        self.name_model = "roberta-base-squad2"
        self.model_path = os.path.join(self.current_dir,self.name_model)
        self.tokenizer = self.__roberta_lib__()
        self.model = pipeline('question-answering', model=self.model_path, tokenizer=self.tokenizer)

    def __roberta_lib__(self):
        tokenizer = RobertaTokenizer.from_pretrained(self.model_path)
        #model = RobertaForQuestionAnswering.from_pretrained( self.model_path)

        return tokenizer
    
    def extract_ans(self):

        for text in self.retrived_ans:
            inputs = self.tokenizer(self.question,text)
            start_positions = torch.tensor([1])
            end_positions = torch.tensor([3])
            transformers_query = {"context": text, "question": self.question}
            #outputs = self.model(**inputs, start_positions=start_positions, end_positions=end_positions)
            #loss = outputs.loss
            #start_scores = outputs.start_logits
            #end_scores = outputs.end_logits
            predictions = self.model(transformers_query,
                                     topk=4,
                                     handle_impossible_answer=True,
                                     max_seq_len=256,
                                     doc_stride=128)
            if type(predictions) == dict:
                predictions = [predictions]
            pass

if __name__ == "__main__":
    docs = ['hello my name is akash , i play basketball, my hobbies are playingm running ,reading, I work at facebook as a ML engineer and I live in india']
    query = "what are akash's hobbies ?"
    rob = roberta(docs=docs,query = query)
    rob.extract_ans()
