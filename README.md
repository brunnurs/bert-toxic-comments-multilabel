# Python-only implementation of the great blogpost "Multi-label Text Classification using BERT – The Mighty Transformer"
Working python implementation. Code of the original Jupyter-Notebook has been splitted into several python files. **Take care, the code quality is quite bad, I might refactor it at a later point**.

Use the files `run_training.py`and `run_prediction.py` to start training/inference.

Make sure you first download the pretrained BERT tensorflow model and convert it to pytorch. That's the only part missing to run the script, all the data is part of this repository.

### Build, Run and Push Docker Image for Training

The docker image in the `docker` folder contains all necessary libraries to train the model.

To build it use the following command:

`docker build -t ursinbrunner/pytorch_bert .`

To run it locally use the following command (assuming your current directory it the project root):

`docker run -v $(pwd):/workspace -it ursinbrunner/pytorch_bert`

afterwards, you can execute any python script with `python run_training.py`

To push it to dockerhub (which you might use to access it from a GPU cloud service) execute:

`docker push ursinbrunner/pytorch_bert:latest`
