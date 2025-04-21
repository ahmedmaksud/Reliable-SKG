import numpy as np
import matplotlib.pyplot as plt
import sys
import math

sys.path.append("../")
from functools import reduce
from tqdm import tqdm
import dask.multiprocessing

dask.config.set(scheduler="processes", num_workers=8)
from dask.diagnostics import ProgressBar

ProgressBar().register()

from CEF_modules import module_SVD_CEF_basic as CEF
from CEF_modules import module_SVD_CEF_advanced as CEFA
from CEF_modules import module_ReedSolo as RS
from CEF_modules import module_ECC_RCHM as ECC


def get_SNRx_set_now():
    return CEFA.get_range(low=10, high=1e7, count=31)


def get_unitary_matrix(N):
    temp = np.random.normal(0, 1, (N, N))
    Q, _ = np.linalg.qr(temp)
    return Q


def random_perm_mat(N):
    I = I = np.eye(N)
    idx = np.random.permutation(N)
    mat = np.zeros((N, N))
    for i in range(N):
        mat[i, :] = I[idx[i], :]
    return mat


def get_perm_mat_set(N, p, m):
    PM = np.zeros((N, N, m, p))
    for i in range(m):
        for j in range(p):
            PM[:, :, i, j] = random_perm_mat(N)
    return PM


def get_KE_DEF(N, keysize, ll, SNRx, partition, midpoint, gc):
    x = CEF.gen_x(N)
    xprime = CEF.add_noise(x, SNRx)
    ind, indprime = CEFA.get_ind_indprime_over(2**ll, x, xprime, partition, midpoint)
    ind = gc[ind]
    indprime = gc[indprime]
    bit = CEFA.get_binary(ind, ll)
    bitprime = CEFA.get_binary(indprime, ll)
    bit = np.ravel(bit)
    bitprime = np.ravel(bitprime)
    bit = bit[0:keysize]
    bitprime = bitprime[0:keysize]
    return np.count_nonzero(ind - indprime) == 0, 1 - (
        np.count_nonzero(bit - bitprime) / keysize
    )


def KE_DEF_main(keysize, N, R, SNRx_set, over, fname):
    print("DEF")
    def_ker = []
    def_ber = []
    for SNRx in tqdm(SNRx_set):
        keysize = int(np.floor((N / 2) * (np.log2(1 + SNRx))))
        ll = int(np.ceil(keysize / N))
        partition = ECC.G_train_y_partition(2 ** (ll + over))
        midpoint = ECC.G_train_y_midpoint(2 ** (ll + over))
        gc = CEFA.grayCode(ll)
        futures = [
            dask.delayed(get_KE_DEF)(N, keysize, ll, SNRx, partition, midpoint, gc)
            for i in range(R)
        ]
        results = dask.compute(futures)[0]
        kk = 0
        bb = 0
        for j in range(R):
            kk += results[j][0]
            bb += results[j][1]
        def_ker.append(1 - (kk / R))
        def_ber.append(1 - (bb / R))
    np.save(
        "data_"
        + fname
        + "/DEF_KER_N"
        + str(N)
        + "kys"
        + str(keysize)
        + "over"
        + str(over),
        np.array(def_ker),
    )
    np.save(
        "data_"
        + fname
        + "/DEF_BER_N"
        + str(N)
        + "kys"
        + str(keysize)
        + "over"
        + str(over),
        np.array(def_ber),
    )
    return


def get_KE_CEF(N, keysize, llprime, SNRx, partition, midpoint, gc, want_prune=True):
    x = CEF.gen_x(N)
    xprime = CEF.add_noise(x, SNRx)
    K = int(np.ceil(keysize / llprime))
    if want_prune:
        Q = CEF.gen_Q(K, x)  # pruned
    else:
        Q = CEF.gen_Q(K, x, eta_thres=math.inf, check_Q=True)  # notpruned
    yy, yyprime = CEFA.get_yy_yyprime(x, xprime, Q)
    ind, indprime = CEFA.get_ind_indprime_over(
        2**llprime, yy, yyprime, partition, midpoint
    )
    ind = gc[ind]
    indprime = gc[indprime]
    bit = CEFA.get_binary(ind, llprime)
    bitprime = CEFA.get_binary(indprime, llprime)
    bit = np.ravel(bit)
    bitprime = np.ravel(bitprime)
    bit = bit[0:keysize]
    bitprime = bitprime[0:keysize]
    return np.count_nonzero(ind - indprime) == 0, 1 - (
        np.count_nonzero(bit - bitprime) / keysize
    )


def KE_SVDCEF_main(keysize, N, R, SNRx_set, llprime, over, fname, want_prune):
    print("SVD-CEF")
    partition = CEFA.train_y_partition(N, 2 ** (llprime + over))
    midpoint = CEFA.train_y_midpoint(N, 2 ** (llprime + over))
    gc = CEFA.grayCode(llprime)
    svd_ker = []
    svd_ber = []
    for SNRx in tqdm(SNRx_set):
        keysize = int(np.floor((N / 2) * (np.log2(1 + SNRx))))
        K = int(np.ceil(keysize / llprime))
        futures = [
            dask.delayed(get_KE_CEF)(
                N, keysize, llprime, SNRx, partition, midpoint, gc, want_prune
            )
            for i in range(R)
        ]
        results = dask.compute(futures)[0]
        kk = 0
        bb = 0
        for j in range(R):
            kk += results[j][0]
            bb += results[j][1]
        svd_ker.append(1 - (kk / R))
        svd_ber.append(1 - (bb / R))
    np.save(
        "data_"
        + fname
        + "/SVD_KER_N"
        + str(N)
        + "kys"
        + str(keysize)
        + "over"
        + str(over)
        + "wantprune"
        + str(want_prune),
        np.array(svd_ker),
    )
    np.save(
        "data_"
        + fname
        + "/SVD_BER_N"
        + str(N)
        + "kys"
        + str(keysize)
        + "over"
        + str(over)
        + "wantprune"
        + str(want_prune),
        np.array(svd_ber),
    )


def get_KE_DRP(N, keysize, llprime, SNRx, partition, midpoint, L, gc, fover):
    K = int(np.ceil(keysize / llprime))
    Q = np.random.normal(0, 1, (L, N, K))
    if K < N:
        Qid = get_unitary_matrix(N)[0:K, :]
    else:
        Qid = get_unitary_matrix(K)[:, 0:N]
    x = CEF.gen_x(N)
    xprime = CEF.add_noise(x, SNRx)
    v = Qid @ x
    vprime = Qid @ xprime
    ppartition = np.sqrt(N / K) * ECC.G_train_y_partition(L * (2**fover))
    mmidpoint = np.sqrt(N / K) * ECC.G_train_y_midpoint(L * (2**fover))
    aaind, aaindprime = CEFA.get_ind_indprime_over(L, v, vprime, ppartition, mmidpoint)
    newQ = np.zeros((K, N))
    newQprime = np.zeros((K, N))
    for i in range(K):
        newQ[i, :] = Q[aaind[i], :, i]
        newQprime[i, :] = Q[aaindprime[i], :, i]
    yy = newQ @ x
    yyprime = newQprime @ xprime
    ind, indprime = CEFA.get_ind_indprime_over(
        2**llprime, yy, yyprime, partition, midpoint
    )
    ind = gc[ind]
    indprime = gc[indprime]
    bit = CEFA.get_binary(ind, llprime)
    bitprime = CEFA.get_binary(indprime, llprime)
    bit = np.ravel(bit)
    bitprime = np.ravel(bitprime)
    bit = bit[0:keysize]
    bitprime = bitprime[0:keysize]
    return np.count_nonzero(ind - indprime) == 0, 1 - (
        np.count_nonzero(bit - bitprime) / keysize
    )


def KE_DRPCEF_main(keysize, N, R, SNRx_set, llprime, over, fname):
    print("DRP-CEF")
    gc = CEFA.grayCode(llprime)
    partition, midpoint = CEFA.train_y_partition_midpoint_RP(N, 2 ** (llprime + over))
    drp_ker = []
    drp_ber = []
    for SNRx in tqdm(SNRx_set):
        keysize = int(np.floor((N / 2) * (np.log2(1 + SNRx))))
        futures = [
            dask.delayed(get_KE_DRP)(
                N, keysize, llprime, SNRx, partition, midpoint, int(N / 2), gc, over
            )
            for i in range(R)
        ]
        results = dask.compute(futures)[0]
        kk = 0
        bb = 0
        for j in range(R):
            kk += results[j][0]
            bb += results[j][1]
        drp_ker.append(1 - (kk / R))
        drp_ber.append(1 - (bb / R))
    np.save(
        "data_"
        + fname
        + "/DRP_KER_N"
        + str(N)
        + "kys"
        + str(keysize)
        + "over"
        + str(over),
        np.array(drp_ker),
    )
    np.save(
        "data_"
        + fname
        + "/DRP_BER_N"
        + str(N)
        + "kys"
        + str(keysize)
        + "over"
        + str(over),
        np.array(drp_ber),
    )


def get_ind_indprime_IOM2(x, xprime, PM, K):
    N, _, m, p = np.shape(PM)
    idx = np.zeros((m, 1))
    idxprime = np.zeros((m, 1))
    for i in range(m):
        v = np.ones((N, 1))
        vprime = np.ones((N, 1))
        for j in range(p):
            v = v * (PM[:, :, i, j] @ x)
            vprime = vprime * (PM[:, :, i, j] @ xprime)
        idx[i, :] = np.argmax(v[0:K, :])
        idxprime[i, :] = np.argmax(vprime[0:K, :])
    return idx.astype(int), idxprime.astype(int)


def get_KE_IOM2(N, Kbits, keysize, SNRx, gc, p=3):
    m = int(np.ceil(keysize / Kbits))
    x = CEF.gen_x(N)
    xprime = CEF.add_noise(x, SNRx)
    PM = get_perm_mat_set(N, p, m)
    ind, indprime = get_ind_indprime_IOM2(x, xprime, PM, 2**Kbits)
    ind = gc[ind]
    indprime = gc[indprime]
    bit = CEFA.get_binary(ind, Kbits)
    bitprime = CEFA.get_binary(indprime, Kbits)
    bit = np.ravel(bit)
    bitprime = np.ravel(bitprime)
    bit = bit[0:keysize]
    bitprime = bitprime[0:keysize]
    return np.count_nonzero(ind - indprime) == 0, 1 - (
        np.count_nonzero(bit - bitprime) / keysize
    )


def KE_IoM_main(keysize, N, R, SNRx_set, fname):
    print("IOM2-CEF")
    gc = CEFA.grayCode(np.ceil(np.log2(N)).astype(int))
    iom_ker = []
    iom_ber = []
    for SNRx in tqdm(SNRx_set):
        keysize = int(np.floor((N / 2) * (np.log2(1 + SNRx))))
        futures = [
            dask.delayed(get_KE_IOM2)(
                N, np.ceil(np.log2(N)).astype(int), keysize, SNRx, gc, p=N
            )
            for i in range(R)
        ]
        results = dask.compute(futures)[0]
        k = 0
        b = 0
        for j in range(R):
            k += results[j][0]
            b += results[j][1]
        iom_ker.append(1 - (k / R))
        iom_ber.append(1 - (b / R))

    np.save(
        "data_"
        + fname
        + "/IOM_KER_N"
        + str(N)
        + "kys"
        + str(keysize)
        + "over"
        + str(over),
        np.array(iom_ker),
    )
    np.save(
        "data_"
        + fname
        + "/IOM_BER_N"
        + str(N)
        + "kys"
        + str(keysize)
        + "over"
        + str(over),
        np.array(iom_ber),
    )


def my_plot2(fname, N):
    DEF_ber0 = np.load("data_" + fname + "/DEF_KER_N16kys186over0.npy")
    DEF_ber1 = np.load("data_" + fname + "/DEF_KER_N16kys186over1.npy")
    DEF_ber3 = np.load("data_" + fname + "/DEF_KER_N16kys186over3.npy")

    SVD_ber0p = np.load("data_" + fname + "/SVD_KER_N16kys186over0wantpruneTrue.npy")
    SVD_ber1p = np.load("data_" + fname + "/SVD_KER_N16kys186over1wantpruneTrue.npy")
    SVD_ber3p = np.load("data_" + fname + "/SVD_KER_N16kys186over3wantpruneTrue.npy")

    SVD_ber0 = np.load("data_" + fname + "/SVD_KER_N16kys186over0wantpruneFalse.npy")
    SVD_ber1 = np.load("data_" + fname + "/SVD_KER_N16kys186over1wantpruneFalse.npy")
    SVD_ber3 = np.load("data_" + fname + "/SVD_KER_N16kys186over3wantpruneFalse.npy")

    SVD_ber1[26:] = 0
    SVD_ber3[26:] = 0

    DRP_ber0 = np.load("data_" + fname + "/DRP_KER_N16kys186over0.npy")
    DRP_ber1 = np.load("data_" + fname + "/DRP_KER_N16kys186over1.npy")
    DRP_ber3 = np.load("data_" + fname + "/DRP_KER_N16kys186over3.npy")

    IOM_ber = np.load("data_" + fname + "/IOM_KER_N16kys186over3.npy")

    SNRx_set = np.load("data_" + fname + "/SNR.npy")

    lw = 0.5
    ms = 5

    ax1 = plt.subplot(1, 1, 1)
    ax1.semilogy(
        10 * (np.log10(SNRx_set)),
        DEF_ber0,
        "-*",
        color="blue",
        linewidth=lw + 0.25,
        label=r"DQ,$l=L_{key}/N$,over=0",
        markersize=ms,
        markevery=slice(0, 37, 3),
    )
    ax1.semilogy(
        10 * (np.log10(SNRx_set)),
        DEF_ber1,
        "--*",
        color="blue",
        linewidth=lw + 0.25,
        label=r"DQ,$l=L_{key}/N$,over=1",
        markersize=ms,
        markevery=slice(1, 37, 3),
    )
    ax1.semilogy(
        10 * (np.log10(SNRx_set)),
        DEF_ber3,
        ":*",
        color="blue",
        linewidth=lw + 0.25,
        label=r"DQ,$l=L_{key}/N$,over=3",
        markersize=ms,
        markevery=slice(2, 37, 3),
    )

    ax1.semilogy(
        10 * (np.log10(SNRx_set)),
        SVD_ber0,
        "-^",
        color="red",
        linewidth=lw + 0.25,
        label=r"SVD-CEF,l'=1,over=0",
        markersize=ms,
        markevery=slice(0, 37, 2),
    )
    ax1.semilogy(
        10 * (np.log10(SNRx_set)),
        SVD_ber1,
        "--^",
        color="red",
        linewidth=lw + 0.25,
        label=r"SVD-CEF,l'=1,over=1",
        markersize=ms,
    )
    ax1.semilogy(
        10 * (np.log10(SNRx_set)),
        SVD_ber3,
        ":^",
        color="red",
        linewidth=lw + 0.25,
        label=r"SVD-CEF,l'=1,over=3",
        markersize=ms,
    )

    ax1.semilogy(
        10 * (np.log10(SNRx_set)),
        SVD_ber0p,
        "-v",
        color="magenta",
        linewidth=lw + 0.25,
        label=r"SVD-CEF pruned,l'=1,over=0",
        markersize=ms,
        markevery=slice(1, 37, 2),
    )
    ax1.semilogy(
        10 * (np.log10(SNRx_set)),
        SVD_ber1p,
        "--v",
        color="magenta",
        linewidth=lw + 0.25,
        label=r"SVD-CEF pruned,l'=1,over=1",
        markersize=ms,
    )
    ax1.semilogy(
        10 * (np.log10(SNRx_set)),
        SVD_ber3p,
        ":v",
        color="magenta",
        linewidth=lw + 0.25,
        label=r"SVD-CEF pruned,l'=1,over=3",
        markersize=ms,
    )

    ax1.semilogy(
        10 * (np.log10(SNRx_set)),
        DRP_ber0,
        "-s",
        color="green",
        linewidth=lw + 0.25,
        label=r"DRP,l'=1,over=0",
        markersize=ms,
        markevery=slice(1, 37, 3),
    )
    ax1.semilogy(
        10 * (np.log10(SNRx_set)),
        DRP_ber1,
        "--s",
        color="green",
        linewidth=lw + 0.25,
        label=r"DRP,l'=1,over=1",
        markersize=ms,
    )
    ax1.semilogy(
        10 * (np.log10(SNRx_set)),
        DRP_ber3,
        ":s",
        color="green",
        linewidth=lw + 0.25,
        label=r"DRP,l'=1,over=3",
        markersize=ms,
    )

    ax1.semilogy(
        10 * (np.log10(SNRx_set)),
        IOM_ber,
        "-o",
        color="#ff7f0e",
        linewidth=lw + 0.25,
        label=r"IoM-2",
        markersize=ms,
        markevery=slice(0, 37, 2),
    )

    ax1.set_ylabel(r"KER")
    ax1.set_xlabel(r"$SNR_x$ (dB)")

    key = []
    SNRx_set = [SNRx_set[i] for i in range(0, len(SNRx_set), 3)]
    for SNRx in SNRx_set:
        key.append(int(np.floor((N / 2) * (np.log2(1 + SNRx)))))

    fig = ax1.figure
    ax1.set_xlim([10, 70])
    ax1.set_ylim([1e-3, 1.2])
    ax1.grid(which="both")
    ax1.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", fontsize=9)

    ax2 = ax1.twiny()
    ax2.set_xticks(10 * np.log10(SNRx_set))
    ax2.set_xticklabels(key)
    ax2.xaxis.set_ticks_position(
        "bottom"
    )  # set the position of the second x-axis to bottom
    ax2.xaxis.set_label_position(
        "bottom"
    )  # set the position of the second x-axis to bottom
    ax2.spines["bottom"].set_position(("outward", 36))
    ax2.set_xlabel("Key Length (bits)")
    ax2.set_xlim(ax1.get_xlim())

    fig.savefig(
        "data_" + fname + "/fig_BER.jpg", bbox_inches="tight", pad_inches=0.02, dpi=450
    )
    fig.show()
    return


# %%
if __name__ == "__main__":
    fname = "data_plots"
    N = 16
    R = 50000
    keysize = 128
    llprime = 1
    SNRx_set = get_SNRx_set_now()
    np.save("data_" + fname + "/SNR", SNRx_set)

    over = 0
    KE_DEF_main(keysize, N, R, SNRx_set, over, fname)
    KE_SVDCEF_main(keysize, N, R, SNRx_set, llprime, over, fname, want_prune=True)
    KE_SVDCEF_main(keysize, N, R, SNRx_set, llprime, over, fname, want_prune=False)
    KE_DRPCEF_main(keysize, N, R, SNRx_set, llprime, over, fname)

    over = 1
    KE_DEF_main(keysize, N, R, SNRx_set, over, fname)
    KE_SVDCEF_main(keysize, N, R, SNRx_set, llprime, over, fname, want_prune=True)
    KE_SVDCEF_main(keysize, N, R, SNRx_set, llprime, over, fname, want_prune=False)
    KE_DRPCEF_main(keysize, N, R, SNRx_set, llprime, over, fname)

    over = 3
    KE_DEF_main(keysize, N, R, SNRx_set, over, fname)
    KE_SVDCEF_main(keysize, N, R, SNRx_set, llprime, over, fname, want_prune=True)
    KE_SVDCEF_main(keysize, N, R, SNRx_set, llprime, over, fname, want_prune=False)
    KE_DRPCEF_main(keysize, N, R, SNRx_set, llprime, over, fname)

    KE_IoM_main(keysize, N, R, SNRx_set, fname)
