import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import seaborn as sns
from scipy import stats
from matplotlib import gridspec
import matplotlib.cm as cm
#from matplotlib import animation
from functools import partial
import multiprocessing as mp
from io_functions import *
from sklearn import preprocessing

def beautify(ax):
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    for line in ax.get_xticklines() + ax.get_yticklines():
        line.set_markersize(0)
        line.set_color("gray")
        line.set_markeredgewidth(1.4)

    for line in ax.xaxis.get_ticklines(minor=True) + ax.yaxis.get_ticklines(minor=True):
        line.set_markersize(0)
    for line in ax.xaxis.get_ticklines(minor=False) + ax.yaxis.get_ticklines(minor=False):
        line.set_markersize(0)

def jointplots(data, save_dir, titles = None, main = "", refcoords = None, refcoords_j=None,
         axes=None, reshape=True, data_j=None, titles_j=None, max_tIC=100, min_density=None, 
         custom_lims=None, custom_lims_j=None, max_diff=2.5, tpt_paths=None, tpt_paths_j=None,
         n_levels=15, worker_pool=None, parallel=True, n_pts=200j, all_apo_data=None, remake=False,
         max_i=10000., min_i=None):
   
  plt.clf()
  print("Making delta G plots.")

  num_columns = np.shape(data)[1]
  jointplot_partial = partial(jointplot, main = main, save_dir = save_dir, titles = titles,
                all_data = data, axes=axes, data_j=data_j, titles_j=titles_j, 
                refcoords=refcoords, refcoords_j=refcoords_j, max_tIC=min(num_columns, max_tIC),
                min_density=min_density, custom_lims=custom_lims, custom_lims_j=custom_lims_j, max_diff=max_diff, tpt_paths=tpt_paths,
                tpt_paths_j=tpt_paths_j, n_levels=n_levels, n_pts=n_pts, all_apo_data=all_apo_data, remake=remake, min_i=min_i)
  i_j_tuples = []
  if data_j is None:
    for i in range(0, data.shape[1]):
      for j in range(0, data.shape[1]):
        if i == j: continue 
        i_j_tuples.append((i, j))
  else:
    for i in range(0, data.shape[1]):
      for j in range(0, data_j.shape[1]):
        i_j_tuples.append((i, j))
        i_j_tuples.append((j,i))

  if parallel and worker_pool is None:
    pool = mp.Pool(int(mp.cpu_count()/2))
    pool.map(jointplot_partial, i_j_tuples)
    pool.terminate()
  elif worker_pool is not None:
    matplotlib.use('Agg')
    worker_pool.map_sync(jointplot_partial, i_j_tuples)
  else:
    for i_j_tuple in i_j_tuples:
      jointplot_partial(i_j_tuple)

  print("Done plotting columns")
  return

def jointplot(i_j_tuple, all_data, save_dir=None, make_animation=False, trajectory=None, video_file=None, titles=None,
              main="", include_1d_kde=False, custom_lims=None, custom_lims_j=None, axes=None, data_j=None, titles_j=None, refcoords=None, 
              refcoords_j=None, max_tIC=5, min_density=None, max_diff=2.5, tpt_paths=None, tpt_paths_j=None, n_levels=15, n_pts=200j,
              all_apo_data=None, remake=False, min_i=None, superpose_circles=None, cmap=None):
  matplotlib.use('Agg')
  i, j = i_j_tuple
  if 1==1:
  #try:
    if data_j is not None: 
      data = np.column_stack([all_data[:,i], data_j[:,j]])
      partial_titles = [titles[i], titles_j[j]]
      if refcoords is not None:
        ref_data = np.column_stack([refcoords[:,i], refcoords_j[:,j]])
      if tpt_paths is not None:
        tpt_data = [np.column_stack([tpt_paths[k][:,i], tpt_paths_j[k][:,j]]) for k in range(0,len(tpt_paths))]
    else:
      data = all_data[:,[i,j]]
      if titles is not None:
        partial_titles = [titles[i], titles[j]]
      else:
        partial_titles = ["", ""]
      if refcoords is not None:
        ref_data = refcoords[:, [i,j]]
      if tpt_paths is not None:
        tpt_data = [tpt_path[:, [i,j]] for tpt_path in tpt_paths]
      if all_apo_data is not None:
        apo_data = all_apo_data[:,[i,j]]
        apo_values = apo_data[:,].T
        apo_kde = stats.gaussian_kde(apo_values, bw_method='silverman')

    if save_dir is not None:
      fig_file = "%s/%s_%s_%s.eps" %(save_dir, main, partial_titles[0], partial_titles[1])
      if os.path.exists(fig_file) and remake is False:
        return 
    else:
      fig_file = None


    print("hi")
    values = data[:,].T
    print(np.shape(values))

    kde1 = stats.gaussian_kde(data[:,0])
    kde2 = stats.gaussian_kde(data[:,1])
    kde = stats.gaussian_kde(values, bw_method='silverman')

    print("Computed 2D KDE")

    # Create a regular 2D grid with 50 points in each dimension
    xmin, ymin = data.min(axis=0) #+ np.array([-.5,.5])
    xmax, ymax = data.max(axis=0) #+ np.array([-.5,.5])
    xmin -= np.std(data[:,0])*1.
    ymin -= np.std(data[:,1])*1.
    xmax += np.std(data[:,0])*1.
    ymax += np.std(data[:,1])*1.

    if custom_lims_j is not None:
      xmin, xmax = custom_lims[i][0], custom_lims[i][1]
      ymin, ymax = custom_lims_j[j][0], custom_lims_j[j][1]
    elif custom_lims is not None:
      xmin, xmax = custom_lims[i][0], custom_lims[i][1]
      ymin, ymax = custom_lims[j][0], custom_lims[j][1]

    xi, yi = np.mgrid[xmin:xmax:n_pts, ymin:ymax:n_pts]

    # Create two 1D grid with 50 points in each dimension
    x = np.linspace(0.8*xmin, 1.2*xmax,100)    
    y = np.linspace(0.8*ymin,1.2*ymax,100)
    dx = kde1(x)
    dy = kde2(y)

    # Evaluate the KDE on a regular grid...
    coords = np.vstack([item.ravel() for item in [xi, yi]])
    density = kde(coords).reshape(xi.shape)
    if all_apo_data is not None:
      apo_density = apo_kde(coords).reshape(xi.shape)
      density -= apo_density
      min_density = np.min(density)
    else:
      density = -0.6 * np.log(density)
      if min_density is None:
        density -= density.min()
        min_density = np.min(density)
        print("min_density=%s" %str(min_density))
    print("Computed Density. Now plotting.")

    #Set grid
    gs = gridspec.GridSpec(2, 2, width_ratios=[3,1], height_ratios=[1,3])

    #Contour Plot
    fig = plt.figure()
    
    #fig.suptitle('My Contour Plot')
    #ax = plt.subplot()
    #ax = plt.axes(xlim=(0,15), ylim=(0,3))
    if include_1d_kde:
      ax = plt.subplot(gs[1,0])
    else:
      ax = fig.add_subplot(111)
    
    """
    if custom_xlim is None:
      print(density)
      print(density.shape)
      xi[np.argmin(np.abs(2.4-density))], 
      xlim = [0.75*xmin, 1.25*xmax]
      ylim = [0.75*ymin, 1.25*ymax]
    else:
      xlim = custom_xlim
      ylim = custom_ylim
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    """
    if titles is not None:
      ax.set_xlabel(partial_titles[0])
      ax.set_ylabel(partial_titles[1])
    if main is not None:
      ax.set_title(main)
    
    if all_apo_data is None:
      vmin = min_density# - (0.2*np.abs(np.std(density)))
      vmax= min_density + max_diff
      print((vmin,vmax))
    else:
      vmin = min_density# - 0.2 * np.abs(np.std(density))
      vmax = np.max(density)# + 0.2 * np.abs(np.max(density))

    if all_apo_data is not None:
      import matplotlib.colors as colors
      pcm = ax.pcolormesh(xi, yi, density,
                       norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,
                                              vmin=vmin, vmax=vmax),
                       cmap='RdBu_r')
    else:
      if cmap is None:
        cmap = plt.cm.get_cmap("coolwarm")
      cmap.set_under([cmap(k) for k in range(cmap.N)][0])
      cmap.set_over([cmap(k) for k in range(cmap.N)][-1])

      cax = ax.contourf(density.T, origin='lower', aspect='auto',cmap=cmap, extent=(xmin, xmax, ymin, ymax), vmin=vmin, vmax=vmax, levels=np.linspace(vmin, vmax, n_levels), extend="min")  
      ax.contour(density.T, origin='lower', aspect='auto',cmap=cm.bone, extent=(xmin, xmax, ymin, ymax), vmax=vmax, linewidths=(1.,), levels=np.linspace(vmin, vmax, n_levels))
    

      cbar = plt.colorbar(cax)
    #ax.contour(density.T, extent = (xmin,xmax,ymin,ymax))#, origin='lower', aspect='auto', cmap=cm.bone)  
    
    if refcoords is not None:
      for i in range(0, ref_data.shape[0]):
        ax.scatter([ref_data[i,0]], [ref_data[i,1]], marker = 'v', c='k',s=75)

    #Marginals
    
    if include_1d_kde:
      axr = plt.subplot(gs[1,1], sharey=ax,xticks=[],xlim=(0,dy.max()),ylim=(ymin,ymax))
      cbar = plt.colorbar(cax)#, ticks=[0.001, .006,.011])
      #cbar.ax.set_yticklabels(['low', 'med', 'high'])
      axt = plt.subplot(gs[0,0], sharex=ax,frameon=False,yticks=[],xlim=(xmin,xmax),ylim=(0,dx.max()))
      axr.plot(dy,y,color='black')
      axt.plot(x,dx,color='black')
      #axr.fill(dy, y, alpha=.75,color='#5673E0')
      #axt.fill(x,dx, alpha=.75,color='#5673E0')

      #Clean Up
      beautify(axr)
      beautify(axt)

    #if make_animation:
    #  ax.scatter([data[0,0]], [data[0,1]], marker = 's', c='g',s=15)

    if tpt_paths is not None:
      annotation_linspace = np.linspace(ymax - np.std(y), ymax, len(tpt_paths))
      print(tpt_paths)

      color=cm.rainbow(np.linspace(0,1,len(tpt_paths)))
      for p, tpt_path in enumerate(tpt_data):
        path = np.array([(pt[0], pt[1]) for pt in tpt_path])
        print(path)
        plt.plot(path[:,0], path[:,1], c=color[p])
        annotate_xy = (0.9*xmax, annotation_linspace[p])
        plt.annotate("TP %d" %(p+1), xy=annotate_xy, xytext=annotate_xy,size=8, color=color[p])
        #plt.scatter(path[0,0], path[0,1],  marker='D', c=color[p], s=15)
        plt.scatter(path[-1,0], path[-1,1],  marker='x', c=color[p], s=50)
        #plt.scatter(path[1:-1,0], path[1:-1,1], marker='o', c=color[p], s=15)


    if superpose_circles is not None:
      coords = superpose_circles["coords"]
      radii = superpose_circles["radii"]
      scaled_radii = preprocessing.MinMaxScaler().fit_transform(radii)
      color=iter(cm.winter(scaled_radii))
      for i, coord in enumerate(coords):
        radius = np.abs(radii[i])
        c = next(color)
        ax.scatter(coord[0], coord[1], marker = 'o', c=c,s=radius*100)
    if fig_file is not None:
      fig.savefig(fig_file)#, format='svg', dpi=1200)
    else:
      plt.show()

    #pp = PdfPages("%s/%s_%s_%s.eps" %(save_dir, main, partial_titles[0], partial_titles[1]))
    #pp = PdfPages(save_file)
    #plt.xlabel(titles[i])
    #plt.ylabel(titles[j])
    #plt.title(main)
    #pp.savefig()
    #pp.close()
    
    if make_animation:
      line, = ax.plot([], [], lw=0.6)
      scatter = ax.scatter([], [], marker = 's', c='g',s=15)
      def init():
        line.set_data([], [])
        scatter.set_offsets([])
        return line, scatter,

      def animate(i):
        #line = ax.scatter([trajectory[1,0]], [trajectory[1,1]], marker = 's', c='g',s=15)
        line.set_data(trajectory[:i,0], trajectory[:i,1])
        scatter.set_offsets(trajectory[i,:])
        return line, scatter

      anim = animation.FuncAnimation(fig, animate,
                                      init_func=init, 
                                      frames=trajectory.shape[0]-1, 
                                      interval=10,
                                      blit=False)
      anim.save(video_file, fps=30, 
            extra_args=['-vcodec', 'h264', 
                        '-pix_fmt', 'yuv420p'])
  #except:
  else:
    return


def test(filename):
  fig = plt.figure()
  ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
  line, = ax.plot([], [], lw=2)

  # initialization function: plot the background of each frame
  def init():
      line.set_data([], [])
      return line,

  # animation function.  This is called sequentially
  def animate(i):
      x = np.linspace(0, 2, 1000)
      y = np.sin(2 * np.pi * (x - 0.01 * i))
      line.set_data(x, y)
      return line,

  # call the animator.  blit=True means only re-draw the parts that have changed.
  anim = animation.FuncAnimation(fig, animate, init_func=init,
                                 frames=200, interval=20, blit=True)

  # save the animation as an mp4.  This requires ffmpeg or mencoder to be
  # installed.  The extra_args ensure that the x264 codec is used, so that
  # the video can be embedded in html5.  You may need to adjust this for
  # your system: for more information, see
  # http://matplotlib.sourceforge.net/api/animation_api.html
  anim.save(filename)#, fps=30, extra_args=['-vcodec', 'libx264'])


