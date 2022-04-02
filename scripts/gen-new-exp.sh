#!/bin/bash
function usage {
  # expXXX or XXX is expected as the argument.
  # `copy-exp.sh exp001` generates `exp002`.
  cat <<EOF
$(basename ${0}) is a wrapper to execute experiment files
Usage:
  $(basename ${0}) [command] [<options>]
Command:
  experiment number to execute (e.g. exp001 or 001)
Options:
  --help, -h    print this
EOF
}

for OPT in "$@"; do
    case "$OPT" in
    '-h' | '--help')
        usage
        exit 1
        ;;
    esac
done

latest_exp_name=$(ls -r src/exp | grep 'exp' | head -n 1)

if [ ! "`echo $1 | grep 'exp'`" ]; then
  base_exp_name=exp$1
else
  base_exp_name=$1
fi

if [ "`echo $latest_exp_name | grep 'exp'`" ]; then
  latest_exp=$(echo $latest_exp_name | sed -e 's/exp//g')
fi
latest_exp_int=$((10#$latest_exp))
new_exp_int=$((latest_exp_int+1))
while [ ${#new_exp_int} -lt 3 ]
do
  new_exp_int=0$new_exp_int
done
new_exp_name=exp$new_exp_int

if [ ! -d "$PWD/src/exp/$base_exp_name" ]; then
  echo "src/exp/$base_exp_name not exists: skip to copy."
  exit 1
fi
if [ -d "$PWD/src/exp/$new_exp_name" ]; then
  echo "src/exp/$new_exp_name already exists: skip to copy."
  exit 1
fi

echo "Execute: 'cp -r src/exp/$base_exp_name src/exp/$new_exp_name'"
cp -r src/exp/$base_exp_name src/exp/$new_exp_name
