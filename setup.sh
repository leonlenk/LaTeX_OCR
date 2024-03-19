set -e

unzip archive.zip -d inkML_data
rm -r inkML_data/trainData_2012_part1/trainData_2012_part1
rm -r inkML_data/trainData_2012_part2/trainData_2012_part2
rm -r inkML_data/MatricesTrain2014/MatricesTrain
rm -r inkML_data/TrainINKML_2013/TrainINKML
python run_preprocess.py

wget https://zenodo.org/api/records/56198/files-archive
unzip files-archive -d rendered_LaTeX
tar -xvzf rendered_LaTeX/formula_images.tar.gz
rm archive.zip
rm files-archive
