from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Plus
from flask import Flask, request, jsonify
import string
#multi-qa-distilbert-cos-v1 BETTER FOR SEMANTIC SEARCH, but it is alot slower
#paraphrase-multilingual-MiniLM-L12-v2 BEST FOR MULTILINGUal search
#Default is all-MiniLM-L6-v2

app = Flask(__name__)


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
          model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2') 
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
     elif(Language== "ar" or Language== "bg" or Language== "ca" or Language=="cs" or Language=="da" or Language=="de" or Language=="el" or Language=="es" or Language=="et" or Language=="fa" or Language=="fi" or Language=="fr" or Language=="fr-ca" or Language=="gl" or Language=="gu" or Language=="he" or Language=="hi" or Language=="hr" or Language=="hu" or Language=="hy" or Language=="id" or Language=="it" or Language=="ja" or Language=="ka" or Language=="ko" or Language=="ku" or Language=="lt" or Language=="lv" or Language=="mk" or Language=="mn" or Language=="mr" or Language=="ms" or Language=="my" or Language=="nb" or Language=="nl" or Language=="pl" or Language=="pt" or Language=="pt-br" or Language=="ro" or Language=="ru" or Language=="sk" or Language=="sl" or Language=="sq" or Language=="sr" or Language=="sv" or Language=="th" or Language=="tr" or Language=="uk" or Language=="ur" or Language=="vi" or Language=="zh-cn" or Language=="zh-tw"):
          model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2') 
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
     app.run(debug=True)