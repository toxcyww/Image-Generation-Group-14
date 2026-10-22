"""Frozen L2CS gaze features; pinned safe-tensor weights and input gradients."""
from runtime import ROOT
import json
import math
import struct
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import Bottleneck
from runtime import load_module


def read_fixed_safetensors(path):
    # This pinned model uses only F32 and I64 tensors. Parse data, never pickle.
    payload=path.read_bytes()
    header_size=struct.unpack('<Q',payload[:8])[0]
    assert 0<header_size<1024*1024
    header=json.loads(payload[8:8+header_size]);offset=8+header_size
    state={}
    types={'F32':torch.float32,'I64':torch.int64}
    for name,entry in header.items():
        if name=='__metadata__':continue
        a,b=entry['data_offsets'];dtype=types[entry['dtype']]
        expected=math.prod(entry['shape'])*torch.empty((),dtype=dtype).element_size()
        assert 0<=a<=b<=len(payload)-offset and b-a==expected
        data=bytearray(payload[offset+a:offset+b])
        state[name]=torch.frombuffer(data,dtype=dtype).reshape(entry['shape']).clone()
    return state


class GazeFeatures(nn.Module):
    def __init__(self,device='cuda'):
        super().__init__()
        module=load_module('_new_l2cs_model',ROOT/'third_party/l2cs/model.py')
        self.net=module.L2CS(Bottleneck,[3,4,6,3],90)
        self.net.load_state_dict(read_fixed_safetensors(ROOT/'models/l2cs_gaze360_resnet50.safetensors'),strict=True)
        self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])
        self.register_buffer('bins',torch.arange(90).float()*4-180)
        self.eval().requires_grad_(False).to(device)

    @staticmethod
    def grid(face,device='cuda',scale=1.):
        x,y,w,h=map(float,face[:4]);cx=x+w/2;cy=y+h/2
        values=(torch.arange(224,device=device).float()+.5)/224-.5
        yy,xx=torch.meshgrid(values,values,indexing='ij')
        xx=cx+xx*w*scale;yy=cy+yy*h*scale
        return torch.stack(((2*xx+1)/256-1,(2*yy+1)/256-1),-1)[None]

    def raw_angles(self,image,grid):
        crop=F.grid_sample((image+1)/2,grid.to(image),align_corners=False,padding_mode='border')
        # Upstream training/test use Resize(448), despite some mirror cards saying 224.
        crop=F.interpolate(crop,(448,448),mode='bilinear',align_corners=False)
        first,second=self.net((crop-self.mean)/self.std)
        # Checkpoint heads are fc_yaw_gaze / fc_pitch_gaze. Flip probes confirm
        # that the first head predominantly changes sign horizontally. The
        # upstream training/pipeline local variable names are inconsistent.
        yaw=(first.softmax(-1)*self.bins).sum(-1)*math.pi/180
        pitch=(second.softmax(-1)*self.bins).sum(-1)*math.pi/180
        return torch.stack((pitch,yaw),-1)

    def forward(self,image,grid):
        mirrored=grid.flip(2).clone();mirrored[...,0]=-mirrored[...,0]
        angles=self.raw_angles(torch.cat((image,image.flip(-1))),torch.cat((grid,mirrored)))
        a,b=angles.chunk(2)
        pitch=(a[:,0]+b[:,0])/2
        yaw=(a[:,1]-b[:,1])/2
        angles=torch.stack((pitch,yaw),-1)
        vector=torch.stack((-pitch.cos()*yaw.sin(),-pitch.sin(),-pitch.cos()*yaw.cos()),-1)
        return angles,vector
