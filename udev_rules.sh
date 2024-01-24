#!/usr/bin/bash

out=/etc/udev/rules.d/99-csi-dissertation.rules

rm -f $out
echo  "ATTRS{idProduct}==\"0030\", ATTRS{idVendor}==\"80ee\", ATTRS{serial}==\"f7892f85bae48874\", SYMLINK+=\"top_camera\"" >> $out