# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from ml_collections import ConfigDict

config = ConfigDict()
config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
