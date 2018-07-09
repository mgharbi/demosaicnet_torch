inspect:
	python bin/inspect_model.py output/simple_kp_bayer_large data/real_mosaic/bayer output/kernel_viz --offset_x 1 --offset_y 0 \
		--shift_x 1024 --shift_y 1024

inspect_xtrans:
	python bin/inspect_model.py output/simple_kp_xtrans data/real_mosaic/xtrans output/kernel_viz_xtrans --offset_x 0 --offset_y 0 \

inspect_data:
	python bin/view_data.py data/images/test/filelist.txt --xtrans

eval_bayer_kpae:
	python bin/eval.py output/kpae_bayer_l2 data/real_mosaic/bayer output/eval/bayer --offset_x 1 \
	--pretrained pretrained_models/bayer

eval_bayer_kp:
	python bin/eval.py output/kp_bayer data/real_mosaic/bayer output/eval/bayer --offset_x 1 \
	--pretrained pretrained_models/bayer

eval_bayer_nn:
	python bin/eval.py output/nn data/real_mosaic/bayer output/eval/bayer --offset_x 1 \
	--pretrained pretrained_models/bayer

eval_xtrans:
	python bin/eval.py output/simple_kp_xtrans data/real_mosaic/xtrans output/eval/simple_kp_xtrans \
	--pretrained pretrained_models/xtrans --xtrans

# train
train_simple_kp_bayer:
	python bin/train.py data/images/train/filelist.txt output/simple_kp_bayer \
		--params model=SimpleKP mosaic_period=2 --loss l2\
		--pretrained pretrained_models/bayer \
		--val_data data/images/val/filelist.txt --batch_size 4 --lr 1e-5

train_simple_kp_bayer_large:
	CUDA_VISIBLE_DEVICES=1 python bin/train.py data/images/train/filelist.txt output/simple_kp_bayer_large \
		--params model=SimpleKP mosaic_period=2 ksize=11 levels=5 --loss l2 \
		--pretrained pretrained_models/bayer \
		--val_data data/images/val/filelist.txt --batch_size 4 --lr 1e-5

train_simple_kp_xtrans:
	CUDA_VISIBLE_DEVICES=1 python bin/train.py data/images/train/filelist.txt output/simple_kp_xtrans \
		--params model=SimpleKP mosaic_period=6 --loss l2 \
		--pretrained pretrained_models/xtrans --xtrans \
		--val_data data/images/val/filelist.txt --batch_size 4 --lr 1e-5

train_simple_kp_xtrans_no_norm:
	CUDA_VISIBLE_DEVICES=0 python bin/train.py data/images/train/filelist.txt output/simple_kp_xtrans_no_norm \
		--params model=SimpleKP mosaic_period=6 normalize=False --loss l2 \
		--pretrained pretrained_models/xtrans --xtrans \
		--val_data data/images/val/filelist.txt --batch_size 4 --lr 1e-4

train_simple_kp_xtrans_large_go:
	CUDA_VISIBLE_DEVICES=1 python bin/train.py data/images/train/filelist.txt output/simple_kp_xtrans_large_go \
		--params model=SimpleKP mosaic_period=6 ksize=15 width=64 activation=relu --loss l2 \
		--pretrained pretrained_models/xtrans --xtrans --green_only \
		--val_data data/images/val/filelist.txt --batch_size 1 --lr 1e-4

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
		--params model=BayerLog --loss vgg \
		--val_data data/images/val/filelist.txt --batch_size 4 --lr 1e-4

train_kp_bayer:
	python bin/train.py data/images/train/filelist.txt output/kp_bayer \
		--params model=BayerKP --loss vgg \
		--pretrained pretrained_models/bayer \
		--val_data data/images/val/filelist.txt --batch_size 4 --lr 1e-4

train_kpae_bayer:
	python bin/train.py data/images/train/filelist.txt output/kpae_bayer \
		--params model=BayerKP autoencoder=True --loss vgg \
		--pretrained pretrained_models/bayer \
		--val_data data/images/val/filelist.txt --batch_size 4 --lr 1e-5

train_kpae_bayer_l2:
	python bin/train.py data/images/train/filelist.txt output/kpae_bayer_l2 \
		--params model=BayerKP autoencoder=True --loss l2 \
		--pretrained pretrained_models/bayer \
		--val_data data/images/val/filelist.txt --batch_size 16 --lr 1e-5

train_kpae_bayer_small:
	python bin/train.py data/images/train/filelist.txt output/kpae_bayer_small \
		--params model=BayerKP autoencoder=True convs=1 levels=3 width=32 ksize=5 --loss l2 \
		--pretrained pretrained_models/bayer \
		--val_data data/images/val/filelist.txt --batch_size 16 --lr 1e-5

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
