#!/bin/bash
alkaid convert "$@" 2>&1 | tee compile.log
