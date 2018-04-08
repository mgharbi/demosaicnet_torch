train_demo:
	python bin/train.py demo_data/filelist.txt output/bayer \
	--pretrained pretrained_models/bayer \
	--val_data demo_data/filelist.txt --batch_size 1

train_bayer:
	python bin/train.py data/images/train/filelist.txt output/bayer_ref \
	--pretrained pretrained_models/bayer \
	--val_data data/images/train/filelist.txt --batch_size 1

train_exp:
	python bin/train.py data/images/train/filelist.txt output/exp \
		--params model=BayerExperimental \
		--val_data data/images/val/filelist.txt --batch_size 32

train_nn:
	python bin/train.py data/images/train/filelist.txt output/nn \
		--params model=BayerNN \
		--val_data data/images/val/filelist.txt --batch_size 128 --lr 1e-5

train_nn_unnormalized:
	python bin/train.py data/images/train/filelist.txt output/nn_unnormalized \
		--params model=BayerNN normalize=False\
		--val_data data/images/val/filelist.txt --batch_size 64 --lr 1e-4

train_log:
	python bin/train.py data/images/train/filelist.txt output/log \
		--params model=BayerLog \
		--val_data data/images/val/filelist.txt --batch_size 4 --lr 1e-4

# train_bayer:
# 	echo "nothing yet"

setup: build download_data

build:
	pip install -r requirements.txt
	git submodule init
	git submodule update

# Launch a server to visualize training (port 8097)
server:
	python -m 'visdom.server'

download_data: data
	cd data && wget https://data.csail.mit.edu/graphics/demosaicnet/download_dataset.py && \
		python download_dataset.py

data:
	mkdir - p data
