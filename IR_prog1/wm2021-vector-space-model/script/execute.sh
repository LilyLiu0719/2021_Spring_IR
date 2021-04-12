#!/bin/bash
# Put your command below to execute your program.
# Replace "./my-program" with the command that can execute your program.
# Remember to preserve " $@" at the end, which will be the program options we give you.

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        -r)
        rocchio=1
        shift # past argument
        ;;
        -i)
        query="$2"
        shift # past argument
        shift # past value
        ;;
        -o)
        ranked="$2"
        shift # past argument
        shift # past value
        ;;
        -m)
        model="$2"
        shift # past argument
        shift # past value
        ;;
        -d)
        doc="$2"
        shift # past argument
        shift # past value
        ;;
        *)    # unknown option
        echo "unknown option $1"
        exit 1
        ;;
    esac
done

if [[ "$rocchio" == 1 ]]; then
    python main.py --model "$model" --query "$query" --ranked "$ranked" --doc "$doc" --rocchio
else
    python main.py --model "$model" --query "$query" --ranked "$ranked" --doc "$doc"
fi

exit 0
