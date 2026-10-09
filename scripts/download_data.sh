#!/usr/bin/env bash
# Download raw datasets into data/raw/<name>/.
# TODO (weeks 1-3): add the download steps for each dataset.
#   SMD, MSL, SMAP, CreditCard  - public
#   Yahoo Webscope              - needs an access request
#   iTrust (SWaT, WADI)         - needs an access request
set -euo pipefail

mkdir -p data/raw/{smd,msl,smap,creditcard}
echo "Fill in download steps in scripts/download_data.sh"
