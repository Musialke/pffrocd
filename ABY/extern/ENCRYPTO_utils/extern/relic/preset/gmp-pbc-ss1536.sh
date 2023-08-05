#!/bin/sh
cmake -DCHECK=off -DARITH=gmp -DBN_PRECI=1536 -DFP_PRIME=1536 -DFP_QNRES=on -DFP_METHD="BASIC;COMBA;COMBA;MONTY;LOWER;LOWER;SLIDE" -DFPX_METHD="INTEG;INTEG;LAZYR" -DPP_METHD="LAZYR;OATEP" -DCFLAGS="-O2 -funroll-loops -fomit-frame-pointer" $1
