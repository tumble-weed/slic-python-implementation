import dutils
dutils.init()
import math
from skimage import io, color
import numpy as np
from tqdm import trange
from collections import defaultdict
import wandb
import faiss.contrib.torch_utils
import faiss
import copy
class SLICProcessor(torch.nn.Module):

    def __init__(self, feats, im_t, K, M, 
    c_weight = 10, 
    s_weight = 1, #float('inf')
    device='cpu',USE_NORM_STD=True,debug=True):
        super().__init__()
        self.c_weight = c_weight
        self.s_weight = s_weight
        self.USE_NORM_STD = USE_NORM_STD
        #device = feats.device
        feats = feats.to(device)
        im_t = im_t.to(device)
        self.debug = debug
        #Y,X = torch.meshgrid(
        #    torch.arange(feats.shape[-2]),
        #    torch.arange(feats.shape[-1]))
        Y,X = torch.meshgrid(
            torch.linspace(0,1,feats.shape[-2]),
            torch.linspace(0,1,feats.shape[-1]))
       
        Y = Y[None,None].float()
        X = X[None,None].float()
        X = X.to(device)
        Y = Y.to(device)
        #self.register_buffer('Y',Y)
        #p46()
        self.USE_STD = True
        self.K = K
        self.M = M
        data = feats
        data = data.to(device) 
        #self.register_buffer('data',data)
        #self.register_buffer('im_t',im_t)
        for name in ['X','Y','data','im_t']:
            self.register_buffer(name,locals()[name])
        #self.to(device)
        #===================================================================
        #===================================================================
        #self.im_t = im_t
        self.image_height = self.data.shape[-2]
        self.image_width = self.data.shape[-1]
        self.N = self.image_height * self.image_width
        self.S = int(math.sqrt(self.N / self.K))

        W = 4*self.S
        def get_std_of_1_channel(x,W):
            square_x = (x ** 2)
            E_x = torch.nn.functional.conv2d(x, (1/(W*W)) * torch.ones(1,1,W,W,device=x.device).float())
            square_of_E_x = E_x**2
            E_square_x =  torch.nn.functional.conv2d(square_x, ( 1/(W*W)) * torch.ones(1,1,W,W,device=x.device).float())
            s = (E_square_x - square_of_E_x).sqrt().mean()
            if not (s>=0):
                p46()
            mu = E_x
            return s, mu
        feat_stds = []
        feat_mus = []
        for dix in range(feats.shape[1]):
            x = self.data[:,dix:dix+1,...]
            s,mu_windowed = get_std_of_1_channel(x,W)
            mu = mu_windowed.mean()
            feat_stds.append(s)
            feat_mus.append(mu)
        feat_std = torch.stack(feat_stds,dim=0) 
        feat_mu = torch.stack(feat_mus,dim=0)
        X_std,Xmu_windowed = get_std_of_1_channel(self.X,W) 
        Y_std,Ymu_windowed = get_std_of_1_channel(self.Y,W) 
        spat_std = torch.stack([Y_std,X_std],dim=0)
        spat_mu = torch.stack([Ymu_windowed.mean(),Xmu_windowed.mean()],dim=0)
        avg_std = (spat_std.sum() +feat_std.sum())/(spat_std.shape[0] + feat_std.shape[0])
        #norm_feat_var = feat_var/avg_var
        #norm_spat_var = spat_var/avg_var
        norm_feat_std = feat_std/avg_std
        norm_spat_std = spat_std/avg_std
        if not self.USE_NORM_STD:
            #norm_feat_var = torch.ones_like(norm_feat_var)
            #norm_spat_var = torch.ones_like(norm_spat_var)
            norm_feat_std = torch.ones_like(norm_feat_std)
            norm_spat_std = torch.ones_like(norm_spat_std)

        #self.register_buffer('feat_var',feat_var)
        #self.register_buffer('spat_var',spat_var)
        #self.register_buffer('avg_var',avg_var)
        #self.register_buffer('feat_mu',feat_mu)
        #self.register_buffer('spat_mu',spat_mu)
        #self.register_buffer('feat_std',torch.sqrt(feat_var))
        #self.register_buffer('spat_std',torch.sqrt(spat_var))
        for name in [
            #'feat_var','spat_var','avg_var',
            'feat_mu','spat_mu',
            'feat_std','spat_std',
            #'norm_feat_var','norm_spat_var',
            'norm_feat_std','norm_spat_std'
            ]:
            self.register_buffer(name,locals()[name])
        #p46()
        #self.clusters = []
        #self.label = {}
        #self.dis = np.full((self.image_height, self.image_width), np.inf)
        self.dis = torch.nn.Parameter(np.inf * torch.ones(1,1,self.image_height,self.image_width, device =device).float())
        label = (-1 * torch.ones(1,1,self.image_height,self.image_width, device = device).long())
        self.register_buffer('label',label)
        self.trends= defaultdict(list)

    #def init_clusters(self):
        clusters = []
        clusters_for_visualize = []
        h = self.S / 2
        w = self.S / 2
        while h < self.image_height:
            while w < self.image_width:
                #self.clusters.append(self.make_cluster(h, w))
                '''
               self.data[h][w][0],
               self.data[h][w][1],
               self.data[h][w][2])
                '''
                #clusters.append(torch.stack( [torch.tensor(h,device=device),torch.tensor(w,device=device),
                #self.data[0,0,int(h),int(w)],self.data[0,1,int(h),int(w)],self.data[0,2,int(h),int(w)]]))
                #=====================================================================================
                #spat = (torch.tensor([h,w],device=device) - self.spat_mu) / self.norm_spat_std
                spat = (torch.tensor([h/(self.image_height-1),w/(self.image_width-1)],device=device))
                assert (spat >= 0).all()
                assert (spat <=1).all()
                spat = spat/self.norm_spat_std
                #f = (self.data[0,:,int(h),int(w)] - self.feat_mu)/self.norm_feat_std
                f = (self.data[0,:,int(h),int(w)])/self.norm_feat_std
                #=====================================================================================
                clusters.append(torch.concatenate([spat,f],dim=0))
                clusters_for_visualize.append(torch.concatenate([spat,self.im_t[0,:,int(h),int(w)]],dim=0))
                #clusters.append(torch.stack( [torch.tensor(h,device=device),torch.tensor(w,device=device),
                #self.data[0,0,int(h),int(w)],self.data[0,1,int(h),int(w)],self.data[0,2,int(h),int(w)]]))
                w += self.S
            w = self.S / 2
            h += self.S
        #clusters = np.array(clusters)
        #clusters = torch.tensor(clusters).float()
        clusters = torch.stack(clusters,dim=0)
        #self.register_buffer('clusters',clusters)
        clusters_for_visualize = torch.stack(clusters_for_visualize,dim=0)
        normalized_data = self.data/self.norm_feat_std[None,:,None,None]
        normalized_X = self.X/self.norm_spat_std[None,1:2,None,None]
        normalized_Y = self.Y/self.norm_spat_std[None,0:1,None,None]
        if False:
            print('check if the std is the same')
            get_std_of_1_channel(normalized_X,W)
            get_std_of_1_channel(normalized_Y,W)
            get_std_of_1_channel(normalized_data[:,:1],W)
            get_std_of_1_channel(normalized_data[:,1:2],W)
        #p46()
        for name in ['clusters','clusters_for_visualize','normalized_data','normalized_Y','normalized_X']:
            self.register_buffer(name,locals()[name])
        self.to(device)

    #def get_gradient(self, h, w):
    #    if w + 1 >= self.image_width:
    #        w = self.image_width - 2
    #    if h + 1 >= self.image_height:
    #        h = self.image_height - 2

    #    gradient = self.data[h + 1][w + 1][0] - self.data[h][w][0] + \
    #               self.data[h + 1][w + 1][1] - self.data[h][w][1] + \
    #               self.data[h + 1][w + 1][2] - self.data[h][w][2]
    #    return gradient

    #def move_clusters(self):
    #    #TODO: this hasnt been converted to torch yet
    #    for cluster in self.clusters:
    #        cluster_gradient = self.get_gradient(cluster.h, cluster.w)
    #        for dh in range(-1, 2):
    #            for dw in range(-1, 2):
    #                _h = cluster.h + dh
    #                _w = cluster.w + dw
    #                new_gradient = self.get_gradient(_h, _w)
    #                if new_gradient < cluster_gradient:
    #                    cluster.update(_h, _w, self.data[_h][_w][0], self.data[_h][_w][1], self.data[_h][_w][2])
    #                    cluster_gradient = new_gradient

    def assignment_faiss(self):
        new_label = self.label.clone()
        new_dis = self.dis.clone()
        #...................................................................
        res = faiss.StandardGpuResources() 
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = 0
     

        gpu_index = faiss.GpuIndexFlatL2(res,self.clusters.shape[-1],cfg)
        #concat_std = torch.cat([torch.sqrt(self.spat_var),torch.sqrt(self.feat_var)],dim=0)
        
        #concat_mu = torch.cat([self.spat_mu,self.feat_mu],dim=0)
        #normalized_clusters = self.clusters/concat_std[None,:]
        gpu_index.add(self.clusters)
       
        #...................................................................
        n_nearest = 2
        normalized_concat_features = torch.cat([self.normalized_Y,self.normalized_X,self.normalized_data],dim=1)
        def flatten_for_faiss(t):
            return torch.permute(t,(0,2,3,1)).reshape(-1,concat_features.shape[1])
        def unflatten(t):
            return torch.permute(t.view(concat_features.shape[0],concat_features.shape[2],concat_features.shape[3],-1),(0,3,1,2))
        normalized_concat_features_flat = flatten_for_faiss(normalized_concat_features)
        #concat_features_flat = torch.permute(concat_features,(0,2,3,1)).reshape(-1,concat_features.shape[1])
        #normalized_concat_features_flat = concat_features_flat/concat_std[None,:]
        D_nearest_clusters, I_nearest_clusters = gpu_index.search( normalized_concat_features_flat.contiguous(),k=n_nearest)# n_nearest,1,H,W
        normalized_Y_nearest_clusters = self.clusters[I_nearest_clusters[:,0]][:,0]
        normalized_X_nearest_clusters = self.clusters[I_nearest_clusters[:,0]][:,1]
        normalized_Y_nearest_clusters = unflatten(normalized_Y_nearest_clusters) 
        normalized_X_nearest_clusters = unflatten(normalized_X_nearest_clusters) 
        #Y_nearest_clusters = torch.permute(Y_nearest_clusters.view(concat_features.shape[0],concat_features.shape[2],concat_features.shape[3],1),(0,3,1,2))
        #X_nearest_clusters = torch.permute(X_nearest_clusters.view(concat_features.shape[0],concat_features.shape[2],concat_features.shape[3],1),(0,3,1,2))
        p46()
        Ymin = (self.Y - 2*self.S).clamp(0,None)
        Ymax = (self.Y + 2*self.S).clamp(None,self.image_height)

        Xmin = (self.X - 2*self.S).clamp(0,None)
        Xmax = (self.X + 2*self.S).clamp(None,self.image_width)
        assert (Y_nearest_clusters >= Ymin).all()
        assert (Y_nearest_clusters <= Ymax).all()

        assert (X_nearest_clusters >= Xmin).all()
        assert (X_nearest_clusters <= Xmax).all()
        self.dis.data.copy_(unflatten(D_nearest_clusters[:,0]))
        self.label.data.copy_(unflatten(I_nearest_clusters[:,0]))
        p46()
#        for cluster_ix,cluster in enumerate(self.clusters):
#            #------------------------------------------------------
#            top = max(cluster[0] - 2 * self.S,0)
#            bottom = min(cluster[0] + 2 * self.S,self.image_height)
#            left = max(cluster[1] - 2 * self.S,0)
#            right = min(cluster[1] + 2 * self.S,self.image_width)
#            #------------------------------------------------------
#            #p46()
#            if True:
#                L,A,B = self.data[:,0:1], self.data[:,1:2], self.data[:,2:3]
#                '''
#                Dc0 = torch.sqrt(
#                    (L - cluster[2])**2 +
#                    (A - cluster[3])**2 +
#                    (B - cluster[4])**2)
#                Dc = torch.sqrt(
#                    ((self.data - cluster[2:][None,:,None,None])**2).sum(dim=1,keepdim=True)
#                )
#                '''
#                Dc0 = torch.cat([
#                    (L - cluster[2])**2,
#                    (A - cluster[3])**2,
#                    (B - cluster[4])**2],dim=1)
#                #p46()
#                fbatch = 64
#                Dc2_normalized = torch.zeros_like(self.data)
#                for bi in range((self.data.shape[1] + fbatch - 1) // fbatch):
#                    d = ((self.data[:,bi*fbatch:(bi+1)*fbatch] - cluster[2 + bi*fbatch: 2+(bi+1)*fbatch][None,:,None,None])**2)
#                    if bi == 0:
#                        if not torch.allclose(d[:,:3],Dc0):
#                            p46()
#                    Dc2_normalized[:,bi*fbatch:(bi+1)*fbatch] = d/(self.feat_var[bi*fbatch:(bi+1)*fbatch][None,:,None,None]/self.avg_var)                
#            '''
#            Ds = torch.sqrt(
#                (self.Y - cluster[0])**2 +
#                (self.X - cluster[1])**2)
#            '''
#            Ds2 = torch.cat([
#                (self.Y - cluster[0])**2,
#                (self.X - cluster[1])**2],dim=1)
#
#            Ds2_normalized = Ds2/(self.spat_var[None,:,None,None]/self.avg_var)
#            #p46()
#            if False:
#                D = torch.sqrt((Dc / self.M)** 2 + (Ds / self.S)** 2)
#            else:
#                D = torch.sqrt(
#                    #Dc2/(self.feat_var[None,:,None,None]/self.avg_var) +
#                    Dc2_normalized.sum(dim=1,keepdim=True) + 
#                    Ds2_normalized.sum(dim=1,keepdim=True)
#                )
#            where_update = D < new_dis
#            new_label[where_update] = cluster_ix
#            new_dis[where_update] = D[where_update]
#            #p46()
#           
#        assert (new_label >= 0).all()
#        self.label.data.copy_(new_label)
#        self.dis.data.copy_(new_dis)


    def assignment(self):
        new_label = self.label.clone()
        new_dis = self.dis.clone()
        new_dis_spat = self.dis.clone()
        new_dis_feat = self.dis.clone()
        for cluster_ix,cluster in enumerate(self.clusters):
            #------------------------------------------------------
            cluster_y,cluster_x = (self.image_height-1)* (cluster[0]*self.norm_spat_std[0]),(self.image_width-1)* (cluster[1]*self.norm_spat_std[1])

            #TODO: should this not be self.S//2
            #...................................
            #half_window_height = 2 * self.S
            #half_window_width = 2 * self.S
            half_window_height =  self.S/2
            half_window_width = self.S/2
            #...................................
            top = (self.image_height-1)* (cluster[0]*self.norm_spat_std[0]) - half_window_height
            bottom = (self.image_height-1)* (cluster[0] *self.norm_spat_std[0]) + half_window_height
            left = (self.image_width-1)* (cluster[1] *self.norm_spat_std[1])- half_window_width
            right = (self.image_width-1)* (cluster[1]*self.norm_spat_std[1]) + half_window_width
            top = max( top,0)
            #TODO: image_height - 1?
            bottom = min(bottom,self.image_height)
            left = max(left,0)
            right = min(right,self.image_width)
            top = int(top)
            bottom = int(bottom)
            left = int(left)
            right = int(right)
            window_of_normalized_data = self.normalized_data[:,:,top:bottom,left:right]
            window_of_normalized_Y = self.normalized_Y[:,:,top:bottom,left:right]
            window_of_normalized_X = self.normalized_X[:,:,top:bottom,left:right]
            window_of_new_dis = new_dis[:,:,top:bottom,left:right]
            #------------------------------------------------------
            #mask = torch.zeros_like(new_dis)
            #mask[:,:,int(top):int(bottom),int(left):int(right)] = 1
            cluster_feats = cluster[2:]
            cluster_spat = cluster[:2]

            fbatch = 64
            Dc2_normalized = torch.zeros_like(window_of_normalized_data)
            for bi in range((self.data.shape[1] + fbatch - 1) // fbatch):
                Dc2_normalized[:,bi*fbatch:(bi+1)*fbatch] = ((window_of_normalized_data[:,bi*fbatch:(bi+1)*fbatch] - cluster_feats[bi*fbatch: (bi+1)*fbatch][None,:,None,None])**2)
            Ds2_normalized = torch.cat([
                (window_of_normalized_Y - cluster_spat[0])**2,
                (window_of_normalized_X - cluster_spat[1])**2],dim=1)

            #c_weight = 10 
            #s_weight = 1 #float('inf')
            D = torch.sqrt(
                Dc2_normalized.sum(dim=1,keepdim=True)/self.c_weight + 
                Ds2_normalized.sum(dim=1,keepdim=True)/self.s_weight
            )
            #D[(1-mask).bool()] = float('inf')
            where_update = D < window_of_new_dis
            new_label[:,:,top:bottom,left:right][where_update] = cluster_ix
            new_dis[:,:,top:bottom,left:right][where_update] = D[where_update]
            if True:
                new_dis_spat[:,:,top:bottom,left:right][where_update] = Ds2_normalized.sum(dim=1,keepdim=True)[where_update]
                new_dis_feat[:,:,top:bottom,left:right][where_update] = Dc2_normalized.sum(dim=1,keepdim=True)[where_update]
                #if not (new_label == cluster_ix).sum() >= 1:
                #    p46()
           
        assert (new_label >= 0).all()
        self.label.data.copy_(new_label)
        self.dis.data.copy_(new_dis)
        #dutils.img_save(self.dis,'dis.png')
        #p46()

    def update_cluster(self):
        new_cluster_means = []
        new_cluster_for_visualize_means = []
        for cluster_ix,cluster in enumerate(self.clusters):
            in_cluster = (self.label == cluster_ix).float()
            '''
            ipdb> self.clusters[cluster_ix]
            tensor([18.0000, 18.0000, 65.5807, 29.8132, 35.0211], dtype=torch.float64)
            '''
            n_in_cluster = in_cluster.sum()
            Y_mean = (self.normalized_Y*in_cluster).sum(dim=(0,2,3))/n_in_cluster
            X_mean = (self.normalized_X*in_cluster).sum(dim=(0,2,3))/n_in_cluster
            f_batch_size = 64
            new_feat_means = []
            for bi in range((self.normalized_data.shape[1] + f_batch_size - 1) // f_batch_size):
                #p46()
                slice_f = slice(bi*f_batch_size,(bi+1)*f_batch_size)
                try:
                    #rhs = self.normalized_data[:,slice_f][in_cluster].mean(dim=(0,-2,-1))
                    rhs = (self.normalized_data[:,slice_f] * (in_cluster)).sum(dim=(0,-2,-1))/n_in_cluster
                    new_feat_means.append(rhs)
                    #self.clusters[cluster_ix][fix] = rhs 
                except IndexError:
                    import ipdb;ipdb.set_trace()
            new_feat_means = torch.cat(new_feat_means,dim=0)
            new_cluster_mean = torch.cat([Y_mean,X_mean,new_feat_means],dim=0)
            new_cluster_means.append(new_cluster_mean)

            assert in_cluster.sum() >= 1 

            #new_cluster_for_visualize_mean = torch.cat([self.im_t[:,cix][in_cluster].mean() for cix in range(3)],dim=0)
            new_cluster_for_visualize_feat_mean  = (self.im_t * in_cluster).sum(dim=(0,-1,-2))/n_in_cluster
            new_cluster_for_visualize_mean = torch.cat([Y_mean,X_mean,new_cluster_for_visualize_feat_mean],dim=0)
            new_cluster_for_visualize_means.append(new_cluster_for_visualize_mean)

        new_cluster_for_visualize_means = torch.stack(new_cluster_for_visualize_means,dim=0)
        new_cluster_means = torch.stack(new_cluster_means,dim=0)
        self.clusters.data.copy_(new_cluster_means)
        self.clusters_for_visualize.data.copy_(new_cluster_for_visualize_means)
        if self.clusters.isnan().any():
            p46()

        if self.clusters_for_visualize.isnan().any():
            p46()
    def save_current_image(self, name, i):
        name2 = os.path.splitext(name)[0] + '2.png'
        image_arr = torch.zeros_like(self.im_t)
        image_arr2 = torch.zeros_like(self.im_t)
        for cluster_ix,cluster in enumerate(self.clusters_for_visualize):
            in_cluster = self.label == cluster_ix 
            cluster = copy.deepcopy(cluster)
            cluster[:2] = cluster[:2] * self.norm_spat_std
            cluster[2:] = cluster[2:] #* self.norm_feat_std
            image_arr[:,:1][in_cluster] = cluster[2]
            image_arr[:,1:2][in_cluster] = cluster[3]
            image_arr[:,2:3][in_cluster] = cluster[4]
            image_arr[:,:,(cluster[0]).long().item()*(self.image_height - 1),(cluster[1]).long().item()*(self.image_width - 1)] = 0
            #p46()
            rand_val = torch.rand(1,device=self.im_t.device)
            image_arr2[:,:1][in_cluster] = rand_val
            image_arr2[:,1:2][in_cluster] = rand_val
            image_arr2[:,2:3][in_cluster] = rand_val
        #-----------------------------------
        '''
        image_arr = tensor_to_numpy(image_arr.permute(0,2,3,1))
        image_arr2 = tensor_to_numpy(image_arr2.permute(0,2,3,1))
        image_arr = image_arr[0]
        image_arr2 = image_arr2[0]
        return save_lab_image(name, image_arr),save_lab_image(name2,image_arr2)
        '''
        #-----------------------------------
        try:
            dutils.img_save(image_arr,os.path.realpath(name),use_matplotlib=False)
            dutils.img_save(image_arr2,os.path.realpath(name2),use_matplotlib=False)
        except Exception as e:
            print(e)
            p46()
        #p46()
        name_parts = os.path.split(name)
        labels_path = os.path.join(*name_parts[:-1],f'labels_{i}.png')
        dutils.img_save(self.label,labels_path,use_matplotlib=False)
        dis_path = os.path.join(*name_parts[:-1],f'dis_{i}.png')
        dutils.img_save(self.dis,dis_path,use_matplotlib=False)
        p46()
        return os.path.realpath(name),os.path.realpath(name2)

    def iterate_10times(self):
        #self.init_clusters()
        #self.move_clusters()
        get_savename = lambda t,loop,m=self.M,k=self.K:f'{t}_M{m}_K{k}_loop{loop}'
        for i in trange(dutils.hardcode(n_iter=10)):
            self.assignment()
            #self.assignment2()
            self.update_cluster()
            name = f'torch_results/{get_savename("lenna",i)}.png'
            #p46()
            if self.debug:
                saved_path,saved_path2 = self.save_current_image(name,i)
                #p46()
                wandb.log(dict(
                    approx_image=wandb.Image(saved_path,caption="approx_image"), 
                    cluster_regions=wandb.Image(saved_path2,caption="cluster_regions") ),commit=False)
                self.trends['dis'].append(self.dis.sum().item())
                wandb.log({
                    "iteration":i,
                    "Number of clusters":len(self.clusters),
                    'dis':self.trends['dis'][-1]
                    })
                dutils.save_plot(self.trends['dis'],'match_distance',f'torch_results/{get_savename("match_distance",i)}.png')


def open_image(path):
    """
    Return:
        3D array, row col [LAB]
    """
    rgb = io.imread(path)
    lab_arr = color.rgb2lab(rgb)
    return lab_arr
def save_lab_image(path, lab_arr):
    """
    Convert the array to RBG, then save the image
    :param path:
    :param lab_arr:
    :return:
    """
    p46()
   
    rgb_arr = color.lab2rgb(lab_arr)
    d = os.path.dirname(path)
    os.makedirs(d,exist_ok=True)
    io.imsave(path, rgb_arr)
    path = os.path.realpath(path)
    print(path)
    return path


if __name__ == '__main__':
    #im = open_image('Lenna.png')
    device = 'cuda'
    #path = 'Lenna.png'
    path = os.path.join(os.environ['IMAGENETIMAGESDIR'],'ILSVRC2012_val_00000001.JPEG')
    im = io.imread(path)
    im = color.rgb2lab(im)
    im_t = torch.tensor(im).permute(2,0,1)[None].float()
    feats = im_t
    configs = [
        dict(K=200,M=40),
        dict(K=300,M=40),
        dict(K=500,M=40),
        dict(K=1000,M=40),
        dict(K=200,M=5),
        dict(K=300,M=5),
        dict(K=500,M=5),
        dict(K=1000,M=5)
    ]
    for config in configs:
        wandb.init(project = "slic-segmentation-torch",config=dict(K=config['K'],M=config['M'], path = path))
        p = SLICProcessor(feats,im_t, K=config['K'], M=config['M'],USE_NORM_STD=True,debug=True,device=device)
        p.to(device)
        p.iterate_10times()
        wandb.finish()
        break


