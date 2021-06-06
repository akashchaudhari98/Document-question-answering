import os
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from retriver import BM25
from retriver import DPR


class Question_answering:
    def __init__(self,file_location) -> None:

        self.file_location = file_location
        file_name = self.file_location.split("/")

        if len(file_name) <=1:
             self.file_name = file_name
        else: 
            self.file_name = file_name[-1]

        self.current_dir = os.getcwd()

        
        with open(self.file_location,errors="ignore") as file_:
            __book_data__ = file_.read()

        self.divided_file = self.__para_division__(__book_data__)
        
        self.dpr = DPR(self.divided_file)
        query = " gal gadot to glamour magzine"
        self.dpr.get_top_k_docs(query)
    
    def __para_division__(self, docs) -> list:
        ''' divides a document into paras of size 500 '''
        word_list = word_tokenize(docs)
        count = 0
        para_no = 0
        data_list = {}
        temp_list = []
        articles = ['a','an','the']
        for words in word_list:
            if words not in articles:
                if count != 20:
                    temp_list.append(words.lower())
                    count = count + 1
                else:
                    para_no = para_no + 1
                    temp_list = " ".join(temp_list)
                    temp_list = re.sub(r'\W+', ' ', temp_list)
                    data_list[para_no] = temp_list
                    count = 0
                    temp_list = []

        print("total no of para ", len(data_list))

        return data_list

if __name__ == "__main__":
    loc = r"D:\projects\document QA\Gal Gadot.txt"
    import time
    start = time.time()
    qa = Question_answering(loc)
    print(time.time()- start)
