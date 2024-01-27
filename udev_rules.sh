#!/usr/bin/bash

out=/etc/udev/rules.d/99-csi-dissertation.rules

rm -f $out
echo  SUBSYSTEM == ""SUBSYSTEM==\"video4linux\", "ATTRS{idProduct}==\"0030\", ATTRS{idVendor}==\"80ee\", ATTRS{serial}==\"f7892f85bae48874\", SYMLINK+=\"top_camera\"" >> $out
echo  SUBSYSTEM == ""SUBSYSTEM==\"video4linux\", "ATTRS{idProduct}==\"0030\", ATTRS{idVendor}==\"80ee\", ATTRS{serial}==\"c01e5495eb0612f\", SYMLINK+=\"bottom_camera\"" >> $out