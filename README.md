***This code is outdated and no longer maintained, please use <https://github.com/mgharbi/demosaicnet> instead***

---

Runs on python3 (tested on Anaconda for Ubuntu 14.03).
Other dependencies are listed in requirements.

1. Install and download the data

```
make setup
```

2. Launch a visdom server to monitor training and display a few images

```
make server
```

3. Train a net

```
make train_demo
```

4. Test a net (todo)


Dataset format
--------------

The dataloader assumes the dataset is given as a listing file containing the relative path 
to the images. For example if you have images like:

```
root
├── filelist.txt
├── hdrvdp
│   ├── 000
│   │   ├── 000001.png
│   │   ├── 000002.png
│   │   ├── 000003.png
│   │   ├── 000004.png
│   │   ├── 000005.png
```

filelist.txt should have one path per line as such:

```
hdrvdp/000/000001.png
hdrvdp/000/000002.png
hdrvdp/000/000003.png
hdrvdp/000/000004.png
hdrvdp/000/000005.png
```

etc.
