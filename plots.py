"""
TODO description.

Author: Spencer M. Richards
        Autonomous Systems Lab (ASL), Stanford
        (GitHub: spenrich)
"""

if __name__ == "__main__":
    import numpy as np
    from scipy.stats import beta
    import pickle
    import os
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    import itertools

    plt.rcParams.update({
        'font.family':       'serif',
        'font.serif':        ['Times', 'Times New Roman'],
        'mathtext.fontset':  'cm',
        'font.size':         16,
        'legend.fontsize':   'medium',
        'axes.titlesize':    'medium',
        'lines.linewidth':   2,
        'lines.markersize':  10,
        'errorbar.capsize':  6,
    })

    # FIGURE 2 ###############################################################
    with open('training_data.pkl', 'rb') as file:
        raw = pickle.load(file)
    w_train = raw['w']
    w_min, w_max = raw['w_min'], raw['w_max']
    a, b = raw['beta_params']
    x = np.linspace(0, 1)
    w_train_pdf = w_min + (w_max - w_min)*x
    p_train = beta.pdf(x, a, b) / (w_max - w_min)

    with open(os.path.join('test_results', 'seed=2_M=10.pkl'), 'rb') as file:
        results = pickle.load(file)
    gains = tuple(itertools.product(
        results['gains']['Λ'], results['gains']['K'], results['gains']['P']
    ))
    num_gains = len(gains)
    w_test = results['w']
    w_min, w_max = results['w_min'], results['w_max']
    a, b = results['beta_params']
    w_test_pdf = w_min + (w_max - w_min)*x
    p_test = beta.pdf(x, a, b) / (w_max - w_min)

    _, bins = np.histogram(np.hstack([w_train, w_test]), bins=15)
    fig, ax = plt.subplots(1, 1, dpi=100, figsize=(8, 4))
    ax.plot(w_train_pdf, p_train,
            label=r'$p_\mathrm{train}(w)$', color='tab:blue')
    ax.hist(w_train, density=True, alpha=0.5, bins=bins, color='tab:blue')
    ax.plot(w_test_pdf, p_test,
            label=r'$p_\mathrm{test}(w)$', color='tab:orange')
    ax.hist(w_test, density=True, alpha=0.5, bins=bins, color='tab:orange')
    ax.set_xlabel(r'$w~\mathrm{[m/s]}$')
    ax.set_ylabel(r'sampling probability')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join('figures', 'fig2.pdf'), bbox_inches='tight')
    plt.show()

    # FIGURE 3 ###############################################################
    with open('test_results_single.pkl', 'rb') as file:
        results = pickle.load(file)

    fig = plt.figure(dpi=100, figsize=(8, 7.5))
    grid = plt.GridSpec(2, 2, width_ratios=[1.5, 1], height_ratios=[1, 1],
                        hspace=0.05, wspace=0.4)
    axes = (plt.subplot(grid[:, 0]),
            plt.subplot(grid[0, 1]),
            plt.subplot(grid[1, 1]))

    t = results['ours_meta']['t']
    r = results['ours_meta']['r']
    xr, yr, ϕr = r[:, 0], r[:, 1], r[:, 2]
    axes[0].plot(xr, yr, '--', color='tab:red', lw=4.)

    methods = ('pid', 'lstsq', 'ours', 'ours_meta')
    colors = ('tab:pink', 'tab:orange', 'tab:green', 'tab:blue')
    labels = ('PID', 'ACMRR', 'ours',
              r'ours, $(\Lambda, K, \Gamma) = ('
              r'\Lambda_\mathrm{meta},K_\mathrm{meta},\Gamma_\mathrm{meta})$')

    for method, color in zip(methods, colors):
        q = results[method]['q']
        x, y, ϕ = q[:, 0], q[:, 1], q[:, 2]
        axes[0].plot(x, y, color=color)

        e_norm = np.linalg.norm(results[method]['e'], axis=1)
        axes[1].plot(t, e_norm, color=color)

        u_norm = np.linalg.norm(results[method]['u'], axis=1)
        axes[2].plot(t, u_norm, color=color)

    axes[0].set_xlabel(r'$x~\mathrm{[m]}$')
    axes[0].set_ylabel(r'$y~\mathrm{[m]}$')
    axes[1].get_xaxis().set_ticklabels([])
    axes[1].set_ylabel(r'$\sqrt{\|\tilde{q}\|_2^2+\|\dot{\tilde{q}}\|_2^2}$')
    axes[2].set_xlabel(r'$t~\mathrm{[s]}$')
    axes[2].set_ylabel(r'$\|u\|_2$')

    im_height = 1.
    im_width = 1.5
    im_x0, im_y0 = -0.8, 4.7
    pad = 0.15
    axes[0].text(im_x0, im_y0 + im_height + pad,
                 r'$w = {:.1f}'.format(results['w']) + r'~\mathrm{m/s}$')
    axes[0].imshow(
        plt.imread(os.path.join('figures', 'wind.png')),
        aspect='auto',
        interpolation='none',
        extent=(im_x0, im_x0 + im_width, im_y0, im_y0 + im_height)
    )
    axes[0].set_xlim([-1., 4.2])
    axes[0].set_ylim([-0.2, 6.2])

    handles = [Line2D([0], [0], color=color, label=label)
               for color, label in zip(colors, labels)]
    handles = [Line2D([0], [0], color='tab:red', label='reference',
               linestyle='--', lw=4.)] + handles
    fig.legend(handles=handles, loc='lower center', ncol=2)
    fig.subplots_adjust(bottom=0.24)
    fig.savefig(os.path.join('figures', 'fig3.pdf'), bbox_inches='tight')
    plt.show()

    # FIGURE 4 ###############################################################
    seeds = np.arange(10)
    Ms = np.array([2, 5, 10, 20, 30, 40, 50])
    methods = ('pid', 'lstsq', 'ours')
    colors = ('tab:pink', 'tab:orange', 'tab:green', 'tab:blue')
    labels = ('PID', 'ACMRR', 'ours',
              r'ours, $(\Lambda,K,\Gamma) = ('
              r'\Lambda_\mathrm{meta},K_\mathrm{meta},\Gamma_\mathrm{meta})$')
    metrics = (
        r'$\dfrac{1}{N_\mathrm{test}}'
        r'\sum_{i=1}^{N_\mathrm{test}}\,\mathrm{RMS}(x_i{-}r_i)$',
        r'$\dfrac{1}{N_\mathrm{test}}'
        r'\sum_{i=1}^{N_\mathrm{test}}\,\mathrm{RMS}(u_i)$',
    )

    rms_error = {
        'pid':       np.zeros((num_gains, seeds.size, Ms.size)),
        'lstsq':     np.zeros((num_gains, seeds.size, Ms.size)),
        'ours':      np.zeros((num_gains, seeds.size, Ms.size)),
        'ours_meta': np.zeros((seeds.size, Ms.size)),
    }
    rms_ctrl = {
        'pid':       np.zeros((num_gains, seeds.size, Ms.size)),
        'lstsq':     np.zeros((num_gains, seeds.size, Ms.size)),
        'ours':      np.zeros((num_gains, seeds.size, Ms.size)),
        'ours_meta': np.zeros((seeds.size, Ms.size)),
    }

    for j, seed in enumerate(seeds):
        for m, M in enumerate(Ms):
            filename = os.path.join('test_results',
                                    'seed={}_M={}.pkl'.format(seed, M))
            with open(filename, 'rb') as file:
                results = pickle.load(file)
            for i, _ in enumerate(gains):
                for method in methods:
                    rms_error[method][i, j, m] = np.mean(
                        results[method].ravel()[i]['rms_error']
                    )
                    rms_ctrl[method][i, j, m] = np.mean(
                        results[method].ravel()[i]['rms_ctrl']
                    )
            rms_error['ours_meta'][j, m] = np.mean(
                results['ours_meta']['rms_error']
            )
            rms_ctrl['ours_meta'][j, m] = np.mean(
                results['ours_meta']['rms_ctrl']
            )

    fig, axes = plt.subplots(2, num_gains,
                             dpi=100, figsize=(18, 6), sharex=True)
    axes[0, 0].set_ylabel(metrics[0])
    axes[1, 0].set_ylabel(metrics[1])
    for j, (λ, k, p) in enumerate(gains):
        for method, color in zip(methods, colors):
            axes[0, j].errorbar(Ms, np.mean(rms_error[method][j], axis=0),
                                np.std(rms_error[method][j], axis=0),
                                fmt='-o', color=color)
        axes[0, j].errorbar(Ms, np.mean(rms_error['ours_meta'], axis=0),
                            np.std(rms_error['ours_meta'], axis=0),
                            fmt='-o', color=colors[-1])
        axes[0, j].set_title(
            r'$(\Lambda,K,\Gamma) = ({})$'.format(
                r','.join([r'{}I' if g == 1 else r'{:g}I'.format(g)
                          for g in (λ, k, p)])
            ), pad=7
        )
        axes[0, j].set_xticks(np.arange(0, Ms[-1] + 1, 10))
        axes[0, j].set_xticks(np.arange(0, Ms[-1] + 1, 5), minor=True)

        for method, color in zip(methods, colors):
            axes[1, j].errorbar(Ms, np.mean(rms_ctrl[method][j], axis=0),
                                np.std(rms_ctrl[method][j], axis=0),
                                fmt='-o', color=color)
        axes[1, j].errorbar(Ms, np.mean(rms_ctrl['ours_meta'], axis=0),
                            np.std(rms_ctrl['ours_meta'], axis=0),
                            fmt='-o', color=colors[-1])
        axes[1, j].set_ylim([10.3, 12.7])
        axes[1, j].set_yticks([10.5, 11., 11.5, 12., 12.5])
        axes[1, j].set_xlabel(r'$M$')
        axes[0, j].set_xticks(np.arange(0, Ms[-1] + 1, 10))
        axes[0, j].set_xticks(np.arange(0, Ms[-1] + 1, 5), minor=True)

    axes[0, 0].set_ylim([-0.05, 2.1])
    axes[0, 1].set_ylim([-0.05, 1.6])
    axes[0, 2].set_ylim([-0.05, 0.46])
    axes[0, 3].set_ylim([-0.05, 0.46])

    axes[0, 0].set_yticks([0., 0.5, 1., 1.5, 2.])
    axes[0, 1].set_yticks([0., 0.5, 1., 1.5])
    axes[0, 2].set_yticks([0., 0.1, 0.2, 0.3, 0.4])
    axes[0, 3].set_yticks([0., 0.1, 0.2, 0.3, 0.4])

    handles = [Patch(color=color, label=label)
               for color, label in zip(colors, labels)]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles))
    fig.subplots_adjust(bottom=0.19, hspace=0.1)
    fig.savefig(os.path.join('figures', 'fig4.pdf'), bbox_inches='tight')
    plt.show()
