#variables
data=$(date +'%Y-%m-%dT%H-%M-%S')

path_to_env="/c/Users/ian-g/OneDrive/Documentos/Comunidade DS/clusterização/cluster/Scripts"
path="/c/Users/ian-g/OneDrive/Documentos/Comunidade DS/clusterização"
cd "$path"
"$path_to_env/papermill" src/c9.0-ias-deploy.ipynb "reports/c9.0-ias-deploy_$data.ipynb"
