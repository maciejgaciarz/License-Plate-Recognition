import matplotlib.pylab as plt 

def show_image(image1,type1, image2, type2):
    fig , axes = plt.subplots(ncols=2, figsize = (10,10))
    ax = axes.ravel()

    ax[0].imshow(image1, cmap='gray')
    ax[0].set_title(type1)
    ax[1].imshow(image2, cmap='gray')
    ax[1].set_title(type2)

    for a in ax:
        a.axis('off') 
    
    plt.show()    