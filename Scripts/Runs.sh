for IT in $(seq -w 01 30); do echo python3 Scripts/00_Environment.py Outfile,$IT,10,50,1.0.txt 10 50 1.0;done | sh

R --vanilla < /home/yatess/RLbreeding/RLbreeding/OptimalPath/Scripts/Plot.offspring.R

