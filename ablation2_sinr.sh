for r in {1e-1,1e-2}
do
        for s in {1e-2,3e-3,1e-3,3e-4,1e-4}
        do
            # python assimilate.py -s $s -o $o -r $r -b $b -f results -d era5 --comp=fouriernet --dyn=neuralode --cudaid=0
            # python assimilate.py -s $s -o $o -r $r -b $b -f results -d era5 --comp=pca --dyn=linreg --cudaid=1
            python assimilate2.py --ds=sw --ed=sinr --ld=neuralode --cudaid=0 --sigma-x-b=1e-1 --sigma-xz-b-ratio=$r --n-obs=1024 --mod-sigma=$s
        done
done
