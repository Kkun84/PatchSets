docker build --pull --rm -f "Dockerfile" -t patch_sets:latest "."
docker run -d -p22222:22 -p6006:6006 -p5000:5000 -p8888:8888 --init --rm -it --gpus=all --ipc=host --user=(id -u):(id -g) --name=(basename $PWD) --env TZ=Asia/Tokyo --volume=$PWD:/workspace --volume=$DATASET:/dataset patch_sets:latest fish
docker exec -itd PatchSets mlflow server --default-artifact-root=gs://YOUR_GCS_BUCKET/path/to/mlruns --host=0.0.0.0 --port=5000 & \
docker exec -itd PatchSets tensorboard --logdir=. --host=0.0.0.0 --port=6006 & \
docker exec -itd PatchSets jupyter-lab --no-browser --port=8888 --ip=0.0.0.0 --allow-root --NotebookApp.token=''

docker exec -it PatchSets fish
docker attach PatchSets
