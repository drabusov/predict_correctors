source /cvmfs/aph.gsi.de/modules.sh
module load aph_all

which python
which gcc
WRKDIR=/tmp/drabusov/predict_correctors_77
cd $WRKDIR

python -c "import correction; correction.run(global_index=77)"
cp -r $WRKDIR/results/* /lustre/bhs/drabusov/cluster-testing/2022-02-09/predict_correctors/results/


