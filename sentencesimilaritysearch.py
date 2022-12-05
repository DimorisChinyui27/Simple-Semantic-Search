from sentence_transformers import SentenceTransformer, util
from flask import Flask, request, jsonify

app = Flask(__name__)

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

@app.route("/findsimilarquestions", methods=['POST'])
def GetSimilarQuestions():
     request_data = request.get_json()
     questionDB_dict =list(request_data["allquestionsinDB"])
     inp_question = request_data["userquestion"]
     similarquestionsDB=[]
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
     return jsonify({'similarquestions':similarquestionsDB}) 

if __name__== "__main__":
     app.run()