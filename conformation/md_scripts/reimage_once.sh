#!/bin/bash

echo "parm " $2 >> traj_script.in
echo "trajin " $1".xtc" >> traj_script.in
echo "center origin '!(:POPC | :TIP3 | :SOD | :CLA)'" >> traj_script.in
echo "image origin center" >> traj_script.in
echo "trajout "$1"_reimaged.xtc" >> traj_script.in
echo "go" >> traj_script.in
eval "$AMBERBIN/cpptraj " $2 " 	traj_script.in"