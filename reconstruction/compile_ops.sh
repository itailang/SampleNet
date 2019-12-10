#!/usr/bin/env bash

cd ./external/grouping/

# compile grouping op
sh tf_grouping_compile.sh

cd ../sampling/

# compile sampling op
sh tf_sampling_compile.sh

cd ../structural_losses/

# compile nndistance op
sh tf_nndistance_compile.sh

# compile approxmatch op
sh tf_approxmatch_compile.sh
