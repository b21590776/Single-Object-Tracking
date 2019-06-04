# Single-Object-Tracking
### Author

Ege Berke Balseven

###  Prerequisites
skimage<br/>
numpy<br/>
torch<br/>
torchvision<br/>
matplotlib<br/>
opencv<br/>

python version Python 3.7.1 <br/>
pytorch version 1.0.1<br/>
CUDA 10<br/>
Run this Command:<br/>
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch<br/>

###  Running the Project

train the model
src/train.py

run the test with trained model
rsc/test.py

Parameters for test.py:<br/>
(17) model_weights # path to trained model <br/>
(18) data_directory # path to video frames <br/>
<br/>
Parameters for train.py:<br/>
(20) epochs <br/>
(21) batch_size <br/>
(22) learning_rate <br/>
(23) save_directory # path the model to be saved <br/>
(109) validation videos path <br/>
(159) train videos path <br/>
