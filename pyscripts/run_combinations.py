import itertools
import master

sec_lvls = [80, 112, 128]
mt_algs = [0,1,2]
combinations = itertools.product(sec_lvls, mt_algs)
for sec_lvl, mt_alg in combinations:
    print(f"Running for {sec_lvl=} and {mt_alg=}")
    master.mt_alg = mt_alg
    master.sec_lvl = sec_lvl
    master.run_test()
    print("done")