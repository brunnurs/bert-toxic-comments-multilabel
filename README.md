### Build, Run and Push Docker Image for Training

The docker image in the `docker` folder contains all necessary libraries to train the model.

To build it use the following command:

`docker build -t ursinbrunner/pytorch_bert .`

To run it locally use the following command (assuming your current directory it the project root):

`docker run -v $(pwd):/workspace -it ursinbrunner/pytorch_bert`

afterwards, you can execute any python script with `python run_training.py`

To push it to dockerhub execute:

`docker push ursinbrunner/pytorch_bert:latest`