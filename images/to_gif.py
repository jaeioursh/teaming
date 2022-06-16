import imageio.v2 as imageio
images = []
filenames = []
for i in range(100):
    filenames.append("dummy_fig{}.png".format(i))

for filename in filenames:
    images.append(imageio.imread(filename))

imageio.mimsave('/home/anna/workspaces/AIC/teaming/basic_vis.gif', images)