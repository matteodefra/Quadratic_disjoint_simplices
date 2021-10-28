import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['figure.figsize'] = 11.7,8.27



custom_conf = {'figure.facecolor': 'white',
 'axes.labelcolor': '.15',
 'xtick.direction': 'out',
 'ytick.direction': 'out',
 'xtick.color': '.15',
 'ytick.color': '.15',
 'axes.axisbelow': True,
 'grid.linestyle': '-',
 'text.color': '.15',
 'font.family': ['sans-serif'],
 'font.sans-serif': ['Arial',
  'DejaVu Sans',
  'Liberation Sans',
  'Bitstream Vera Sans',
  'sans-serif'],
 'lines.solid_capstyle': 'round',
 'patch.edgecolor': 'w',
 'patch.force_edgecolor': True,
#  'image.cmap': 'rocket',
 'xtick.top': False,
#  'ytick.right': False,
 'axes.grid': True,
#  'axes.facecolor': '#EAEAF2',
#  'axes.edgecolor': 'white',
 'grid.color': 'lightgrey',
 'axes.spines.left': True,
 'axes.spines.bottom': True,
 'axes.spines.right': True,
 'axes.spines.top': True
#  'xtick.bottom': False,
#  'ytick.left': False
}

# sns.set_theme(style='ticks',rc = {'axes.grid': True})
sns.set_theme(style="ticks", rc = custom_conf)
sns.set_palette('pastel')


df = pd.read_csv("results/n=5000/results_n=5000_K=10_update=3_defl=false_step=0.csv", delimiter=',')


axes = sns.lineplot(
    x='Iteration',
    y='Dual_gap',
    data=df,
    alpha=0.9, 
    legend='full',
    lw=3,
    label=r"""$f(x^*)$ - $\phi (\lambda)$""",
    color= sns.color_palette('pastel')[3]
)

axes.set_xscale('log', base=10)
axes.set_yscale('log', base=10)

axes.set_ylabel(r"""$f(x^*)$ - $\phi (\lambda)$""")

axes.figure.savefig("results/n=5000/n=5000_K=10_gap_rule=3.png")
plt.clf()


axes = sns.lineplot(
    x='Iteration',
    y='DualValue',
    data=df,
    alpha=0.9, 
    legend='full',
    lw=3,
    label=r"""$\phi (\lambda)$""",
    color= sns.color_palette('pastel')[3]
)

axes.ticklabel_format(style='sci', scilimits=(0,0), axis='y')

axes.set_xscale('log', base=10)

axes.set_ylabel(r"""$\phi (\lambda)$""")

axes.figure.savefig("results/n=5000/n=5000_K=10_dual_rule=3.png")
plt.clf()


ax = df.plot(x="Iteration", y="x_norm_residual", legend=False, alpha=0.6, color= sns.color_palette('pastel')[2], lw=3, label=r"""$\parallel x_t$ - $x_{t-1} \parallel$""")
ax2 = ax.twinx()

df.plot(x="Iteration", y="λ_norm_residual", ax=ax2, legend=False, color= sns.color_palette('pastel')[3], alpha=0.6, lw=3, label=r"""$\parallel λ_t$ - $λ_{t-1} \parallel$""")
ax.figure.legend(loc='upper center')

ax.set_xscale('log', base=10)
ax.set_yscale('log', base=10)

ax2.set_xscale('log', base=10)
ax2.set_yscale('log', base=10)

ax2.set_ylabel(r"""$\lambda$ residual""")
ax.set_ylabel(r"""$x$ residual""")

ax.figure.savefig("results/n=5000/n=5000_K=10_lambda_rule=3.png")
