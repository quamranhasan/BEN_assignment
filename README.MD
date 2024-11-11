In this project, I built a chatbot using the REST protocol. The main idea is to utilize Retrieval
Augmentation Generation on a given PDF. 

The model was also tested with multiple prompts and their output and latency were observed. Several chunking techniques
were used to see the effect of segmentation on the performance. The best technique was found out to be 500_75

## How to run
1. Make sure to install the dependencies before running the code
2. The PDF to be analyzed is to be uploaded to the knowledge directory
3. Run embeddings.py to create the embeddings
4. Then run app.py and use a client like POSTMAN.
    ● Method: POST
    ● URL: http://localhost:5000/chat
    ● Header: add content-type: application/json
    ● Feed the query in the Body window