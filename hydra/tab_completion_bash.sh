# source "this_file.sh" "target_python_file.py"
python $1 --help
eval "$(python $1 -sc install=bash)"
