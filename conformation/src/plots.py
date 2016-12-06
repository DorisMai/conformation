import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
plt.style.use('ggplot')
from io_functions import *
import pandas as pd
import seaborn as sns
import multiprocessing as mp
from scipy import stats
from functools import partial

def plot_importances_barh(importances, titles, main, xlabel, ylabel, save_file, n_features=None):
  if n_features is None:
    n_features = len(importances)
  bar_width = 0.2
  opacity = 0.8
  index = np.arange(start=1, stop=n_features+1, step=1)
  plt.barh(titles[:n_features], importances[:n_features], bar_width, alpha=opacity, color='b',label=main)
  plt.ylabel(xlabel)
  plt.xlabel(ylabel)
  plt.title(title)
  plt.xticks(index + bar_width, titles[:n_features], rotation='vertical')
  pp = PdfPages(save_file)
  pp.savefig()
  pp.close()
  plt.clf()

def plot_histogram(j, data, save_dir, main, titles=None):
  column = data[:,j]
  hist = plt.hist(column, 100, normed=True)

  kde = stats.gaussian_kde(column)

  print("Computed 1D KDE")

  x = np.linspace(0.8*column.min(), 1.2*column.max(),1000)       
  dx = kde(x)

  plt.plot(x, dx, lw=2)
  if titles is not None:
    plt.title(titles[j])

  pp = PdfPages("%s/%s_%d.pdf" %(save_dir, main, j+1))
  pp.savefig()
  pp.close()
  plt.clf()


def plot_histograms(data_file, save_dir, main, titles=None):
  data = np.concatenate(load_file(data_file))
  plt.clf()

  plot_histogram_partial = partial(plot_histogram, data=data, save_dir= save_dir, main=main, titles=titles)

  pool = mp.Pool(mp.cpu_count())
  pool.map(plot_histogram_partial, range(0, np.shape(data)[1]))
  pool.terminate()

"""
#Not included yet in Python 3 version of this code.

def plot_corner(data_file, plot_file, title, x_prefix, chosen_columns=None):
  data = np.concatenate(load_file(data_file))
  if chosen_columns is not None:
    data = data[:,chosen_columns]

  ptitles = []
  for j in range(0,np.shape(data)[1]):
    ptitles.append("%s %d" % (x_prefix, j))
  figure = corner.corner(data, labels=ptitles,
                           show_titles=True, title_args={"fontsize": 12}, color='rgb')
  figure.gca().annotate(title, xy=(0.5, 1.0), xycoords="figure fraction",
                        xytext=(0, -5), textcoords="offset points",
                        ha="center", va="top")
  pp = PdfPages(plot_file)
  pp.savefig(figure)
  pp.close()
"""

def plot_seaborn(data_file, plot_file, title, x_prefix, chosen_columns=None):
  data = np.concatenate(load_file(data_file))
  if chosen_columns is not None:
    data = data[:,chosen_columns]

  ptitles = []
  for j in range(0,np.shape(data)[1]):
    ptitles.append("%s %d" % (x_prefix, j))

  df = pd.DataFrame(data, columns=ptitles)
  g = sns.PairGrid(df)
  g.map_diag(sns.kdeplot)
  g.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=6);
  g.savefig(plot_file)

def plot_timescales(operator_file, plot_file, title):
  operator = verboseload(operator_file)
  timescales = operator.timescales_
  print("timescales")
  print(timescales)
  df = pd.DataFrame(columns=['timescales'])
  df['timescales'] = np.array(timescales)
  df.index = list(range(1,len(timescales)+1))
  plt.figure()
  df.plot(kind='bar'); plt.axhline(0, color='k')
  plt.title(title)
  pp = PdfPages(plot_file)
  pp.savefig()
  pp.close()  

def plot_data_vs_column(i, data_1, data_2, names_1, names_2, save_dir):
  for j in range(0,np.shape(data_2)[1]):
    print("Currently analyzing %s, %s" %(names_1[i], names_2[j]))

    tic_i = data_1[:,i]
    tic_j = data_2[:,j]
    plt.hexbin(tic_i, tic_j, bins = 'log', mincnt=1, cmap=plt.cm.RdYlBu_r)
    plt.xlabel(names_1[i])
    plt.ylabel(names_2[j])
    pp = PdfPages("%s/%s_%s_hexbin.pdf" %(save_dir, names_1[i], names_2[j]))
    pp.savefig()
    pp.close()
    plt.clf()  

def plot_data_vs_data(data_1, data_2, names_1, names_2, save_dir, custom_tic_range=None):
  if custom_tic_range is None:
    tic_range = range(0, np.shape(data_1)[1])
  else:
    tic_range = custom_tic_range

  plot_data_vs_column_partial = partial(plot_data_vs_column, data_1=data_1, data_2=data_2, names_1=names_1, names_2=names_2, save_dir=save_dir)
  pool = mp.Pool(mp.cpu_count())
  pool.map(plot_data_vs_column_partial, tic_range)
  pool.terminate()
  return

def plot_heatmap(corr_matrix, row_names, col_names, save_file):
  fig = plt.figure()
  fig.set_size_inches(10. * len(col_names) / len(row_names), 10.) 
  ax = fig.add_subplot(111)
  heatmap = ax.pcolor(corr_matrix, cmap=plt.cm.RdYlBu_r)

  # put the major ticks at the middle of each cell
  ax.set_xticks(np.arange(corr_matrix.shape[1])+0.5, minor=False)
  ax.set_yticks(np.arange(corr_matrix.shape[0])+0.5, minor=False)

  # want a more natural, table-like display
  #ax.invert_yaxis()
  #ax.xaxis.tick_top()

  ax.set_xticklabels(col_names, minor=False, rotation=270)
  ax.set_yticklabels(row_names, minor=False)
  fig.subplots_adjust(bottom=0.2)

  #fig.savefig("%s/bu72_vs_pp_pearson_heatmap.svg" % analysis_dir, format='svg', dpi=1200)
  #plot_file = "%s/bu72_vs_pp_pearson_heatmap.pdf" % analysis_dir
  pp = PdfPages(save_file)

  pp.savefig(fig)

  pp.close()  
  #plt.show()


def plot_clustermap(corr_df, save_file, method='single',
                    row_cluster=True, col_cluster=True,
                    xtick_labelsize=8, ytick_labelsize=8,
                    z_score=0):

  sns.set_style("darkgrid", {"figure.facecolor": "white"})
  plt.rcParams['xtick.labelsize'] = xtick_labelsize
  plt.rcParams['ytick.labelsize'] = ytick_labelsize
  ratio = float(corr_df.shape[0]) / float(corr_df.shape[1])
  if ratio > 1.:
    figsize=(8./ratio , 8.)
  else:
    figsize = (8., 8./ratio)

  #fig = plt.figure(figsize)
  #fig.patch.set_alpha(0.)
  #ax = fig.add_subplot(111)
  g = sns.clustermap(corr_df, z_score=z_score, method=method, row_cluster=row_cluster, col_cluster=col_cluster)
  plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
  sns.set(font_scale=0.5)
  
  g.savefig(save_file, facecolor='w', edgecolor='w')
  #plt.show()