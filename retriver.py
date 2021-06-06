import os 
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from transformers import DPRQuestionEncoder, DPRContextEncoder
from transformers import DPRQuestionEncoderTokenizer,DPRContextEncoderTokenizer

class BM25:

    def __init__(self,documents,method = "BM25") -> None:
        self.passages = documents
        if method == "BM25":
            tokenized_corpus = [doc.split(" ") for doc in self.passages]
            self.bm25 = BM25Okapi(tokenized_corpus)
        #if method == "DPR":
            
        else: raise ValueError("specify a supported method")
        pass
   
    def related_documents(self,query):
        tokenized_query = query.split(" ")
        retrived = self.bm25.get_top_n(tokenized_query, self.passages, n=5)
        for r in retrived:
            print(r)
            print("------------------------------------------------------------------------")
        print(retrived)

class DPR:
    def __init__(self,docs) -> None:
        self.context_documents= docs
        self.current_dir = os.getcwd()
        self.context_encoder_model = "dpr-ctx_encoder-single-nq-base"
        self.question_encoder_model = "dpr-question_encoder-single-nq-base"
        self.context_encoder_model_path = os.path.join(self.current_dir,self.context_encoder_model)
        self.question_encoder_model_path = os.path.join(self.current_dir,self.question_encoder_model)

        self.context_encoder,self.context_tokenizer = self.__context_encoder__()
        self.question_encoder,self.question_tokenizer = self.__question_encoder__()

        self.context_embeddings = self.__encode_context_documents__(self.context_documents)
        
        
    def __context_encoder__(self):
        tokenizer = DPRContextEncoderTokenizer.from_pretrained(self.context_encoder_model_path)

        model = DPRContextEncoder.from_pretrained(self.context_encoder_model_path)

        #input_ids = tokenizer("Hello, is my dog cute ?", return_tensors='pt')["input_ids"]

        #embeddings = model(input_ids).pooler_output

        return model,tokenizer
        pass
    
    def __question_encoder__(self):
        tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(self.question_encoder_model_path)

        model = DPRQuestionEncoder.from_pretrained(self.question_encoder_model_path)

        #input_ids = tokenizer("Hello, is my dog cute ?", return_tensors='pt')["input_ids"]

        #embeddings = model(input_ids).pooler_output
        return model,tokenizer
        pass

    def __encode_context_documents__(self,docs):
        '''encode and store context documents '''
        emmbeddings = {}
        for idx,data in docs.items():
            input_ids = self.context_tokenizer(data,return_tensors='pt')['input_ids']
            emd = self.context_encoder(input_ids).pooler_output
            emmbeddings[idx] = emd
        return emmbeddings

    def get_top_k_docs(self,query,k=5):
        ''' returns top k most similar documents'''
        if query == "" or query == None or not isinstance(query,str):
            raise ValueError("query cannot be empty and must be a string datatype")
        
        input_ids = self.question_tokenizer(query,return_tensors='pt')['input_ids']
        question_emd = self.question_encoder(input_ids).pooler_output
        question_emd = question_emd.detach().numpy()
        scores = {}
        if not isinstance(question_emd, torch.Tensor):
            question_emd = torch.tensor(question_emd)
        
        for key,emd in self.context_embeddings.items():
            if not isinstance(emd, torch.Tensor):
                emd = torch.tensor(emd)

            if len(emd.shape) == 1:
                emd = emd.unsqueeze(0)

            if len(question_emd.shape) == 1:
                question_emd = question_emd.unsqueeze(0)

            score = torch.mm(emd, question_emd.transpose(0, 1))
            scores[key] = score.item()

        top_para = sorted(scores, key=scores.get, reverse=True)[:k]
        top_k = []
        for key in top_para:
            top_k.append(self.context_documents[key])
        
        print(top_k)

        pass

if __name__ == "__main__":
    docs = ["hello", " what is your name", "my name is akash", "hello please go fuck yourself","tum sab ke maa ki ankh"]
    query = "who are you"
    dpr = DPR(docs)
    dpr.get_top_k_docs(query = query)


