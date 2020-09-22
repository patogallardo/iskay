python lups2maggies.py $1
cat ./output/maggies.txt | fit_coeffs > ./output/fitted_coeffs.dat
cat ./output/fitted_coeffs.dat | reconstruct_maggies > ./output/X1.dat
cat ./output/fitted_coeffs.dat | reconstruct_maggies --band-shift 0.1 --redshift 0. > ./output/X2.dat
python AbsMag.py $1
rm ./output/maggies.txt ./output/fitted_coeffs.dat ./output/X1.dat
rm ./output/X2.dat
