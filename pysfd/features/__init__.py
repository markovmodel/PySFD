# This file is part of PySFD.
#
# Copyright (c) 2018 Sebastian Stolzenberg,
# Computational Molecular Biology Group,
# Freie Universitaet Berlin (GER)
#
# for any feedback or questions, please contact the author:
# Sebastian Stolzenberg <ss629@cornell.edu>
#
# PySFD is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##############################################################################

r"""
=======================================
PySFD Features - Significant Feature Differences Analyzer for Python
=======================================

Special Note:
This package uses the pandas module, whose std() function applied
on an array-like of length N uses \sqrt(N - 1) and not
\sqrt(N) - as in numpy - as the denominator.
Therefore, all manual standard deviation implementations in PySFD
consistently use \sqrt(N - 1).
"""

# only necessary for Python 2
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _

from . import srf
from . import prf
from . import spbsf
from . import pspbsf
from . import pprf
from . import pff
