biosemi64.txt is the unchanged montage table shipped with MNE-Python 1.13.2
(mne/channels/data/montages/biosemi64.txt). Its 64 EEG positions are used here;
Nz, LPA and RPA are fiducials and are excluded from the data channels.
MNE's BSD-3-Clause license is retained in LICENSE-MNE.txt.

Theta and Phi are in degrees. The benchmark maps them to a sphere of radius
0.095 m: x=r*sin(theta)*cos(phi), y=r*sin(theta)*sin(phi), z=r*cos(theta).
These are idealized montage coordinates, not participant digitizations.
No MNE installation is needed to run this benchmark.
