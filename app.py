"""Train a small neural pixel classifier on synthetic microscopy scenes and evaluate on separate images."""
from kit import *
TITLE='Neural Microscopy Foreground Segmentation'
LIVE=False

def image(seed,size=24):
    rng=np.random.default_rng(seed);yy,xx=np.mgrid[:size,:size];mask=np.zeros((size,size),bool)
    for _ in range(3):
        cx,cy=rng.uniform(5,size-5,2);radius=rng.uniform(2,4);mask|=(xx-cx)**2+(yy-cy)**2<radius**2
    brightness=rng.uniform(.6,1.1);im=.15+mask*brightness+rng.normal(0,.22,mask.shape);return im,mask.astype(float)
def patches(im):return np.lib.stride_tricks.sliding_window_view(np.pad(im,1,mode='reflect'),(3,3)).reshape(-1,9)
def init():
    rng=np.random.default_rng(7);return [rng.normal(0,.2,(9,8)),np.zeros(8),rng.normal(0,.2,8),np.array(0.)]
def forward(p,x):
    h=np.tanh(x@p[0]+p[1]);prob=1/(1+np.exp(-np.clip(h@p[2]+p[3],-30,30)));return h,prob
def loss_grad(p,x,y):
    h,prob=forward(p,x);loss=-np.mean(y*np.log(prob+1e-12)+(1-y)*np.log(1-prob+1e-12));delta=(prob-y)/len(y);dh=(delta[:,None]*p[2])*(1-h*h)
    return float(loss),[x.T@dh,dh.sum(axis=0),h.T@delta,np.array(delta.sum())]
def iou(y,p):return float(np.sum((y>0)&p)/max(1,np.sum((y>0)|p)))
def run(d,a):
    training=[image(a.seed+i) for i in range(10)];x=np.concatenate([patches(im) for im,y in training]);y=np.concatenate([m.ravel() for im,m in training]);p=init()
    for _ in range(220):
        loss,g=loss_grad(p,x,y)
        for value,grad in zip(p,g):value-=.25*grad
    vi,vy=image(a.seed+50);vp=forward(p,patches(vi))[1];threshold=max(np.arange(.2,.81,.1),key=lambda t:iou(vy.ravel(),vp>=t));ti,ty=image(a.seed+60);prediction=forward(p,patches(ti))[1].reshape(ti.shape)
    return dict(test_iou=iou(ty,prediction>=threshold),intensity_baseline_iou=iou(ty,ti>=.5),selected_threshold=float(threshold),training_images=10,split='Independent whole images for training, validation and test'),dict(**{f'p{i}':v for i,v in enumerate(p)},test_image=ti,test_mask=ty,test_probability=prediction)

if __name__=="__main__":main(run)
