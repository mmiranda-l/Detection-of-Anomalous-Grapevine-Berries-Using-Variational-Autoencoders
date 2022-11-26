

#block(name=job_miro, threads=3, memory=25000, subtasks=1, gpus=1, hours=5)
	echo $CUDA_VISIBLE_DEVICES
    source /home/s0mimira/anaconda3/etc/profile.d/conda.sh
	conda activate annomalie
	python ./utils/train.py 