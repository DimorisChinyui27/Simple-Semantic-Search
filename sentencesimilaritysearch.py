from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Plus
from flask import Flask, request, jsonify
import string

app = Flask(__name__)

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def remove_puncts(input_string, string):
    return input_string.translate(str.maketrans('', '', string.punctuation)).lower()


@app.route("/findsimilarquestions", methods=['POST'])
def GetSimilarQuestions():
     request_data = request.get_json()
     Language=str(request_data["language"]).lower()
     questionDB_dict =list(request_data["allquestionsinDB"])
     inp_question = request_data["userquestion"]
     similarquestionsDB=[]
     if(Language=="en"):
          inp_question_representation = model.encode(inp_question,
          convert_to_tensor=True)

          master_dict_representation = model.encode(questionDB_dict,
          convert_to_tensor=True)

          similarity = util.pytorch_cos_sim(inp_question_representation,
          master_dict_representation )
          _,indices=similarity.sort(descending=True,stable=True)
          indices=indices.numpy()
          indices=indices.flatten()

          for x in indices:
               similarquestionsDB.append(questionDB_dict[x])
     else:
          cleaned_questionDB = []
          tokenized_cleaned_questionDB = []
          for question in questionDB_dict:
               cleaned_question= remove_puncts(question, string)
               cleaned_questionDB.append(cleaned_question)

          for cleanquestion in cleaned_questionDB:
               cleanquestion=cleanquestion.split()
               tokenized_cleaned_questionDB.append(cleanquestion)

          bm25 = BM25Plus(tokenized_cleaned_questionDB)
          tokenized_inp_question=remove_puncts(inp_question, string).split(" ")     
          similarquestionsDB.append(bm25.get_top_n(tokenized_inp_question,questionDB_dict, n=len(questionDB_dict)))
          
     return jsonify({'similarquestions':similarquestionsDB})


if __name__== "__main__":
     app.run()