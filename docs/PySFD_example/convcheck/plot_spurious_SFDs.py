import pandas as pd 
#%matplotlib inline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as _sns; _sns.set()
#from matplotlib.backends.backend_pdf import PdfPages as _PdfPages


for feature_func_name in ["prf.distance.Ca2Ca.std_err", "prf.distance.Ca2Ca.std_dev"]:
    df_tmp = pd.read_csv("df_sdfcomps.%s.dat" % (feature_func_name), sep = "\t", header = [0,1,2])
    #df_tmp.columns = df_tmp.columns.droplevel(2)
    df_tmp.rename(columns = { key: "" for key in [i for i in df_tmp.columns.get_level_values(0) if "Unnamed:" in i ] }, inplace = True)
    df_tmp.rename(columns = { key: "" for key in [i for i in df_tmp.columns.get_level_values(1) if "Unnamed:" in i ] }, inplace = True)
    df_tmp.rename(columns = { key: "" for key in [i for i in df_tmp.columns.get_level_values(2) if "Unnamed:" in i ] }, inplace = True)
    df_tmp[("","","num_bs")]

    plt.figure()
    plt.errorbar(df_tmp[("","","num_bs")], df_tmp[("numsdf","mean","")], yerr = df_tmp[("numsdf","std","")], label = "all SFDs")
    plt.errorbar(df_tmp[("","","num_bs")], df_tmp[("bsandref","mean","")], yerr = df_tmp[("bsandref","std","")], label = r"$\cap$(SFDs, ref.)")
    plt.errorbar(df_tmp[("","","num_bs")], df_tmp[("bsniref","mean","")],  yerr = df_tmp[("bsniref","std","")],  label = r"SFDs \ ref.")
    plt.errorbar(df_tmp[("","","num_bs")], df_tmp[("refnibs","mean","")],  yerr = df_tmp[("refnibs","std","")],  label = r"ref. \ SFDs")
    plt.xlabel("no. of bootstrapped trajectories")
    plt.ylabel("no. of significantly different features")
    plt.title(feature_func_name)
    axes = plt.gca()
    print(axes.get_ylim())
    #axes.set_ylim((0, axes.get_ylim()[1]))
    axes.set_ylim((0, 2000))
    #print(axes.get_ylim())
    plt.legend()
    plt.savefig("df_sdfcomps.%s.pdf" % (feature_func_name))
