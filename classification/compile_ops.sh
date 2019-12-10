#!/usr/bin/env bash

cd ./grouping/

# compile grouping op
sh tf_grouping_compile.sh

cd ../structural_losses/

# compile nndistance op
sh tf_nndistance_compile.sh

# compile approxmatch op
sh tf_approxmatch_compile.sh
