# POSSUM-Semantic-Search-Engine
This search engine takes in a json request containing the question being searched and all the questions in the platform database

It then returns a json object containing a dictionary with key "similarquestions" and value an array that ranks all questions on platform from similarity to our user query question
it uses semantic search when the imput language is "en" and token based search when input language is anything else
