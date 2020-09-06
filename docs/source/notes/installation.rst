Installation
=======================

The framework installation goes as follows.

.. contents::
    :local:

Conda virtual environment setup
--------------------------------
.. code-block:: none

	conda create -y --name icenet python==3.8.5
	conda install -y --name icenet -c conda-forge --file requirements.txt

	conda activate icenet
	* xgboost, pytorch, torch-geometric ... setup now inside the environment *

	...[do your work]...
	conda deactivate

	conda info --envs
	conda list --name icenet


XGBoost setup
--------------
.. code-block:: none

	# Pick CPU or GPU version

	conda install -c conda-forge py-xgboost
	conda install -c nvidia -c rapidsai py-xgboost


Pytorch and torchvision setup
------------------------------

.. code-block:: none

	# Pick CPU or GPU version (check CUDA version with nvidia-smi)

	conda install pytorch==1.5.1 torchvision==0.6.1 cpuonly -c pytorch
	conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2 -c pytorch


Pytorch-geometric setup
--------------------------

.. code-block:: none
	
	# Pick CPU or GPU version

	export CUDA=cpu
	export CUDA=cu102 (or cu92, cu101)

	pip install torch-scatter==latest+$CUDA -f https://pytorch-geometric.com/whl/torch-1.5.0.html
	pip install torch-sparse==latest+$CUDA -f https://pytorch-geometric.com/whl/torch-1.5.0.html
	pip install torch-cluster==latest+$CUDA -f https://pytorch-geometric.com/whl/torch-1.5.0.html
	pip install torch-spline-conv==latest+$CUDA -f https://pytorch-geometric.com/whl/torch-1.5.0.html
	pip install torch-geometric