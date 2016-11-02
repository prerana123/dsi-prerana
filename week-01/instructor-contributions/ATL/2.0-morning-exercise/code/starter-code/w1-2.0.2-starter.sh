#!/bin/bash

echo "Welcome, $USER. to Data Science Immersive"
echo "This is the current directory, $PWD"
echo "These are the files in current directory"

ls

NEWDIR="/Users/mannyglover/Desktop/GeneralAssembly/tmp/bashTest"
mkdir "$NEWDIR"

RUN_PYTHON=true

cd "$NEWDIR"
echo "This is the new current directory, $PWD"
echo "About to make a new python file, print python code to this file, and run this python file..."

touch hello_world.py
echo "print('Hello, world!')" > hello_world.py

#This is to make the Python file executable:
chmod +x hello_world.py

if [ "$RUN_PYTHON" = true ] ; then
	echo "About to call ./hello_world.py"
	python ./hello_world.py
else
	echo "Please set environment variable RUN_PYTHON to true."
fi
