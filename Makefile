train_demo:
	python bin/train.py demo_data/filelist.txt output/bayer \
	--pretrained pretrained_models/bayer \
	--val_data demo_data/filelist.txt --batch_size 1

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
