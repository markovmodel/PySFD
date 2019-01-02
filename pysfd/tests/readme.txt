On behalf of a full unit test, see docs/PySFD_example for an illustrative example.

In docs/PySFD_example, you can run, e.g., via
nohup python PySFD_example.py &> log &
on a Desktop with six freely available CPU cores.

You will likely receive an error message, e.g.,
if VMD or HBPLUS is not properly installed.
If you have properly installed HBPLUS, please update
"export hbdir="
in pysfd/features/scripts/compute_PI.hbplus.sh
, and make this file executable:
chmod +x pysfd/features/scripts/compute_PI.hbplus.sh

The commands
"""
diff -r output/meta output.bak/meta > diff.log
"""
should then result at most in minor numerical differences for
the partial correlation computations.
(For these computations, there are still some numerical hick-ups
 we have to take care of)

Thus the following should result in no significant numerical differences
"""
mkdir bak
mv output/meta/*partial* bak
diff -r output/meta output.bak/meta > diff.log
mv bak/* output/meta
rm -r bak
"""
