# source "this_file.sh" "target_python_file.py"
python $argv[1] --help
python $argv[1] -sc install=fish | source
