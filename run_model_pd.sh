#variables
data=$(date +'%Y-%m-%dT%H-%M-%S')

path="/home/ubuntu/ds-em-clusterizacao"
path_to_env="/home/ubuntu/.pyenv/shims"
cd "$path"
"$path_to_env/papermill" src/c9.0-ias-deploy.ipynb "reports/c9.0-ias-deploy_$data.ipynb"
