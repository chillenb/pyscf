# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

target_sources(corepyscf PRIVATE
  CxLebedevGrid.c
  grid_basis.c
  nr_numint.c
  r_numint.c
  numint_uniform_grid.c
  xc_deriv.c
  nr_numint_sparse.c
  multigrid.c
  grid_common.c
  grid_collocate.c
  grid_integrate.c
)

if(ENABLE_LIBXSMM)
  add_definitions(-DHAVE_LIBXSMM)
endif()

if(ENABLE_LIBXC)
  target_sources(corepyscf PRIVATE libxc_itrf.c)
endif()

if(ENABLE_XCFUN)
  target_sources(corepyscf PRIVATE xcfun_itrf.c)
endif()

target_include_directories(corepyscf PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
