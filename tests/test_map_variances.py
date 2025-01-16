# Calculate Noise and Signal variances

from emda2.core import iotools, fft, plotter
import emda2.emda_methods2 as em
import fcodes2

def compute_variance(hf1, hf2, bin_idx, nbin):
    f_hf1 = fft.to_f(hf1.workarr)
    f_hf2 = fft.to_f(hf2.workarr)
    (
        _,
        _,
        noisevar,
        signalvar,
        totalvar,
        bin_fsc,
        bincount,
    ) = fcodes2.calc_fsc_using_halfmaps(
        f_hf1,
        f_hf2,
        bin_idx,
        nbin,
        0,
        f_hf1.shape[0],
        f_hf1.shape[1],
        f_hf1.shape[2],
    )
    return signalvar, noisevar

def read_map(imap):
    m1 = iotools.Map(imap)
    m1.read()
    return m1

if __name__=="__map__":
    path = "/home/uab58757_local/Dataprocessing/G3_all/data_for_white_paper/"
    halfmap1list = [
        "all_half1.mrc",
        "1200_half1.mrc",
        "600_half1.mrc",
        "300_half1.mrc",
    ]

    nvlist = []
    svlist = []

    for i, half1 in enumerate(halfmap1list):
        hf1 = read_map(path+half1)
        hf2 = read_map(path+half1.replace("half1", "half2"))

        # Create bin_idx
        if i == 0:
            nbin, res_arr, bin_idx, sgrid = em.get_binidx(hf1.workcell, hf1.workarr)

        sv, nv = compute_variance(hf1, hf2, bin_idx, nbin)
        
        svlist.append(sv)
        nvlist.append(nv)

    # plot variances
    print("Plotting variances...")
    plotter.plot_nlines_log(
        res_arr=res_arr,
        list_arr=nvlist,
        labels=["all", "1200", "600", "300"],
        plot_title="noise variances(free-ribosomes)",
        ylabel="log(noise variance)",
        mapname="noise_variances.eps",
    )